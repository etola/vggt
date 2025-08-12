#!/usr/bin/env python3

import argparse
import os
import numpy as np
import trimesh
from PIL import Image
import torch
import torch.nn.functional as F
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from utils.colmap_utils import load_colmap_calibration


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate point cloud from single depth map")
    parser.add_argument("--scene_dir", type=str, required=True,
                       help="Path to scene directory containing images/ and data_subdir/ with VGGT outputs")
    parser.add_argument("--idx", type=int, required=True,
                       help="Index of the image to generate point cloud for (0-based)")
    parser.add_argument("--conf_threshold", type=float, default=2.0,
                       help="Confidence threshold for filtering points (default: 2.0)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for point cloud (default: scene_dir/pointclouds)")
    parser.add_argument("--vggt_model_resolution", type=int, default=518,
                       help="VGGT model resolution (default: 518)")
    parser.add_argument("--data_subdir", type=str, default="out_clean",
                       help="Subdirectory containing vggt data (default: out_clean)")
    parser.add_argument("--use_colmap", action="store_true", default=False,
                       help="Use COLMAP calibration instead of individual camera parameters")
    parser.add_argument("--colmap_subdir", type=str, default="colmap_calibration/batch_000",
                       help="COLMAP calibration subdirectory (default: colmap_calibration/batch_000)")
    return parser.parse_args()


def find_image_files(images_dir):
    """Find all image files in the images directory."""
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for filename in sorted(os.listdir(images_dir)):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    return image_files


def find_depth_confidence_files(raw_data_dir):
    """Find all depth and confidence map files in the raw_data directory."""
    if not os.path.exists(raw_data_dir):
        raise ValueError(f"Raw data directory not found: {raw_data_dir}")
    
    depth_files = []
    confidence_files = []
    
    for filename in sorted(os.listdir(raw_data_dir)):
        if filename.endswith('_depth.npy'):
            base_name = filename[:-10]  # Remove '_depth.npy'
            depth_files.append((base_name, filename))
        elif filename.endswith('_confidence.npy'):
            base_name = filename[:-15]  # Remove '_confidence.npy'
            confidence_files.append((base_name, filename))
    
    if not depth_files:
        raise ValueError(f"No depth map files found in {raw_data_dir}")
    
    if not confidence_files:
        raise ValueError(f"No confidence map files found in {raw_data_dir}")
    
    return depth_files, confidence_files


def load_calibration_data(calibration_dir):
    """Load COLMAP calibration data."""
    if not os.path.exists(calibration_dir):
        raise ValueError(f"Calibration directory not found: {calibration_dir}")
    
    calibration_data = load_colmap_calibration(calibration_dir)
    if not calibration_data or 'images' not in calibration_data:
        raise ValueError(f"Could not load calibration from {calibration_dir}")
    
    return calibration_data


def convert_colmap_intrinsics_to_vggt_format(colmap_intrinsic, depth_width, depth_height, original_image_path=None):
    """
    Convert COLMAP intrinsics to VGGT format by scaling based on resolution differences.
    
    COLMAP often stores normalized intrinsics. VGGT expects pixel-space intrinsics with 
    focal lengths in pixels for the depth map resolution.
    
    Args:
        colmap_intrinsic: 3x3 intrinsic matrix from COLMAP
        depth_width: Width of the depth map in pixels (e.g., 518)
        depth_height: Height of the depth map in pixels (e.g., 518) 
        original_image_path: Path to original image to get its resolution
    
    Returns:
        3x3 intrinsic matrix in VGGT format for depth map resolution
    """
    # Extract focal lengths and principal point from COLMAP intrinsics
    fx_colmap = colmap_intrinsic[0, 0]
    fy_colmap = colmap_intrinsic[1, 1] 
    cx_colmap = colmap_intrinsic[0, 2]
    cy_colmap = colmap_intrinsic[1, 2]
    
    print(f"🔍 COLMAP intrinsics: fx={fx_colmap:.6f}, fy={fy_colmap:.6f}, cx={cx_colmap:.6f}, cy={cy_colmap:.6f}")
    
    # Get original image resolution if available
    if original_image_path and os.path.exists(original_image_path):
        from PIL import Image
        with Image.open(original_image_path) as img:
            orig_width, orig_height = img.size
        print(f"📏 Original image resolution: {orig_width}x{orig_height}")
        print(f"📏 Depth map resolution: {depth_width}x{depth_height}")
    else:
        # Assume square original image with reasonable resolution
        orig_width = orig_height = 2048  # Common camera resolution
        print(f"⚠️  Original image not found, assuming {orig_width}x{orig_height}")
    
    # Check if intrinsics are normalized (typical signs: fx/fy around 1.0, cx/cy around 0.0)
    if abs(fx_colmap - 1.0) < 0.1 and abs(fy_colmap - 1.0) < 0.1:
        print("⚠️  Detected normalized intrinsics, converting to pixel space...")
        
        # For normalized intrinsics, we need to scale them to the depth map resolution
        # Normalized intrinsics assume focal length of 1.0 for a "unit" image
        # We need to scale this to actual pixel focal lengths for the depth map resolution
        
        # Calculate scale factors from original image to depth map
        scale_x = depth_width / orig_width
        scale_y = depth_height / orig_height
        
        print(f"📐 Scale factors: x={scale_x:.6f}, y={scale_y:.6f}")
        
        # Estimate reasonable focal lengths for the original image
        # Based on analysis of actual VGGT intrinsics, focal lengths are typically ~1.66x image dimensions
        fx_orig_estimate = orig_width * 1.66  # Corresponds to ~35° horizontal FOV
        fy_orig_estimate = orig_height * 1.65  # Corresponds to ~35° vertical FOV
        
        # Scale to depth map resolution  
        fx_vggt = fx_orig_estimate * scale_x
        fy_vggt = fy_orig_estimate * scale_y
        cx_vggt = depth_width / 2.0
        cy_vggt = depth_height / 2.0
        
        print(f"🔧 Converted to VGGT format: fx={fx_vggt:.6f}, fy={fy_vggt:.6f}, cx={cx_vggt:.6f}, cy={cy_vggt:.6f}")
    else:
        # Intrinsics appear to be in pixel space already
        # Check if they're for the original image resolution or depth map resolution
        # VGGT intrinsics for 518x518 are typically around 850-900 pixels focal length
        if fx_colmap > depth_width * 2.0:  # Likely for original image (much larger than depth map)
            print("⚠️  Intrinsics appear to be for original image, scaling to depth map resolution...")
            
            # Scale from original image to depth map resolution
            scale_x = depth_width / orig_width
            scale_y = depth_height / orig_height
            
            fx_vggt = fx_colmap * scale_x
            fy_vggt = fy_colmap * scale_y
            cx_vggt = cx_colmap * scale_x if cx_colmap > 0 else depth_width / 2.0
            cy_vggt = cy_colmap * scale_y if cy_colmap > 0 else depth_height / 2.0
            
            print(f"📐 Scale factors: x={scale_x:.6f}, y={scale_y:.6f}")
            print(f"🔧 Scaled to depth map resolution: fx={fx_vggt:.6f}, fy={fy_vggt:.6f}, cx={cx_vggt:.6f}, cy={cy_vggt:.6f}")
        else:
            # Already at depth map resolution (VGGT intrinsics are typically 800-900 for 518x518)
            fx_vggt = fx_colmap
            fy_vggt = fy_colmap
            cx_vggt = cx_colmap if cx_colmap > 0 else depth_width / 2.0
            cy_vggt = cy_colmap if cy_colmap > 0 else depth_height / 2.0
            
            print(f"✅ Using COLMAP intrinsics as-is (already at depth map resolution): fx={fx_vggt:.6f}, fy={fy_vggt:.6f}, cx={cx_vggt:.6f}, cy={cy_vggt:.6f}")
    
    # Construct VGGT-format intrinsic matrix
    vggt_intrinsic = np.array([
        [fx_vggt, 0, cx_vggt],
        [0, fy_vggt, cy_vggt],
        [0, 0, 1]
    ])
    
    return vggt_intrinsic


def generate_single_pointcloud(scene_dir, idx, conf_threshold=2.0, vggt_model_resolution=518, output_dir=None, 
                             data_subdir="out_clean", use_colmap=False, colmap_subdir="colmap_calibration/batch_000"):
    """
    Generate point cloud for a single depth map.
    
    Args:
        scene_dir: Path to scene directory
        idx: Index of the image to process (0-based)
        conf_threshold: Confidence threshold for filtering points
        vggt_model_resolution: VGGT model resolution
        output_dir: Output directory for point cloud
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Setup paths
    images_dir = os.path.join(scene_dir, "images")
    raw_data_dir = os.path.join(scene_dir, data_subdir, "raw_data")
    individual_cameras_dir = os.path.join(scene_dir, data_subdir, "individual_cameras")
    colmap_calibration_dir = os.path.join(scene_dir, data_subdir, colmap_subdir)
    
    if output_dir is None:
        output_dir = os.path.join(scene_dir, "pointclouds")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔍 Scene directory: {scene_dir}")
    print(f"📷 Processing image index: {idx}")
    print(f"📁 Images directory: {images_dir}")
    print(f"🗂️  Raw data directory: {raw_data_dir}")
    if use_colmap:
        print(f"🎯 COLMAP calibration directory: {colmap_calibration_dir}")
    else:
        print(f"🎯 Individual cameras directory: {individual_cameras_dir}")
    print(f"💾 Output directory: {output_dir}")
    
    try:
        # Find image files
        image_files = find_image_files(images_dir)
        if idx >= len(image_files):
            raise ValueError(f"Image index {idx} is out of range. Found {len(image_files)} images.")
        
        target_image_name = image_files[idx]
        print(f"🎯 Target image: {target_image_name}")
        
        # Find depth and confidence files
        depth_files, confidence_files = find_depth_confidence_files(raw_data_dir)
        
        # Create mapping from base names to files
        depth_map = {base: filename for base, filename in depth_files}
        confidence_map = {base: filename for base, filename in confidence_files}
        
        # Find matching depth and confidence files for the target image
        base_name = os.path.splitext(target_image_name)[0]
        
        if base_name not in depth_map:
            raise ValueError(f"No depth map found for image {target_image_name}")
        if base_name not in confidence_map:
            raise ValueError(f"No confidence map found for image {target_image_name}")
        
        depth_file = os.path.join(raw_data_dir, depth_map[base_name])
        confidence_file = os.path.join(raw_data_dir, confidence_map[base_name])
        
        print(f"🗺️  Depth map: {depth_file}")
        print(f"📊 Confidence map: {confidence_file}")
        
        # Load camera parameters (either individual or from COLMAP)
        if use_colmap:
            # Load from COLMAP calibration
            calibration_data = load_calibration_data(colmap_calibration_dir)
            
            # Find camera data for the target image
            if target_image_name not in calibration_data['images']:
                raise ValueError(f"No calibration data found for image {target_image_name} in COLMAP")
            
            camera_data = calibration_data['images'][target_image_name]
            extrinsic = camera_data['extrinsic']  # Shape: [3, 4]
            colmap_intrinsic = camera_data['intrinsic']  # Shape: [3, 3]
            
            # Convert COLMAP intrinsics to VGGT format  
            original_image_path = os.path.join(images_dir, target_image_name)
            intrinsic = convert_colmap_intrinsics_to_vggt_format(
                colmap_intrinsic, vggt_model_resolution, vggt_model_resolution, original_image_path
            )
        else:
            # Load individual camera parameters
            from utils.colmap_utils import load_individual_camera_parameters
            camera_data = load_individual_camera_parameters(target_image_name, individual_cameras_dir)
            if camera_data is None:
                raise ValueError(f"No camera parameters found for image {target_image_name}")
            
            extrinsic = camera_data['extrinsic']  # Shape: [3, 4]
            intrinsic = camera_data['intrinsic']  # Shape: [3, 3]
        
        print(f"📐 Camera extrinsic shape: {extrinsic.shape}")
        print(f"📐 Camera intrinsic shape: {intrinsic.shape}")
        
        # Load depth and confidence maps
        depth_map = np.load(depth_file)
        confidence_map = np.load(confidence_file)
        
        print(f"🗺️  Depth map shape: {depth_map.shape}")
        print(f"📊 Confidence map shape: {confidence_map.shape}")
        
        # Load original image for color sampling
        image_path = os.path.join(images_dir, target_image_name)
        if not os.path.exists(image_path):
            print(f"⚠️  Warning: Original image not found at {image_path}, using gray colors")
            image_array = None
        else:
            image = Image.open(image_path).convert('RGB')
            depth_h, depth_w = depth_map.shape
            image_resized = image.resize((depth_w, depth_h), Image.Resampling.LANCZOS)
            image_array = np.array(image_resized)
            print(f"🎨 Loaded image for color sampling: {image_array.shape}")
        
        # Filter by confidence
        conf_mask = confidence_map > conf_threshold
        print(f"📊 Confidence filtering: {np.sum(conf_mask)}/{depth_map.size} points above threshold {conf_threshold}")
        
        if np.sum(conf_mask) == 0:
            print("❌ No points above confidence threshold!")
            return False
        
        # Unproject depth map to 3D points
        depth_map_batch = depth_map[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
        extrinsic_batch = extrinsic[np.newaxis, ...]
        intrinsic_batch = intrinsic[np.newaxis, ...]
        
        points_3d_batch = unproject_depth_map_to_point_map(depth_map_batch, extrinsic_batch, intrinsic_batch)
        points_3d = points_3d_batch[0]  # Remove batch dimension
        
        # Filter points by confidence
        filtered_points = points_3d[conf_mask]
        
        # Sample colors
        if image_array is not None:
            filtered_colors = image_array[conf_mask]
        else:
            # Fallback to gray colors
            filtered_colors = np.zeros((len(filtered_points), 3), dtype=np.uint8)
            filtered_colors[:, :] = 128
        
        print(f"🎯 Generated {len(filtered_points)} points from depth map")
        
        # Create and save point cloud
        point_cloud = trimesh.PointCloud(vertices=filtered_points, colors=filtered_colors)
        output_path = os.path.join(output_dir, f"{base_name}_pointcloud.ply")
        point_cloud.export(output_path)
        
        print(f"💾 Point cloud saved to: {output_path}")
        print(f"📊 Point range: X[{filtered_points[:,0].min():.3f}, {filtered_points[:,0].max():.3f}], "
              f"Y[{filtered_points[:,1].min():.3f}, {filtered_points[:,1].max():.3f}], "
              f"Z[{filtered_points[:,2].min():.3f}, {filtered_points[:,2].max():.3f}]")
        print(f"🎨 Color range: R[{filtered_colors[:,0].min()}, {filtered_colors[:,0].max()}], "
              f"G[{filtered_colors[:,1].min()}, {filtered_colors[:,1].max()}], "
              f"B[{filtered_colors[:,2].min()}, {filtered_colors[:,2].max()}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating point cloud: {e}")
        return False


def main():
    args = parse_arguments()
    
    print("🌤️  Single Depth Map Point Cloud Generator")
    print("=" * 50)
    
    success = generate_single_pointcloud(
        scene_dir=args.scene_dir,
        idx=args.idx,
        conf_threshold=args.conf_threshold,
        vggt_model_resolution=args.vggt_model_resolution,
        output_dir=args.output_dir,
        data_subdir=args.data_subdir,
        use_colmap=args.use_colmap,
        colmap_subdir=args.colmap_subdir
    )
    
    if success:
        print("\n✅ Point cloud generation completed successfully!")
    else:
        print("\n❌ Point cloud generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main() 