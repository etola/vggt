#!/usr/bin/env python3

import argparse
import os
import numpy as np
import trimesh
from PIL import Image
import torch
import torch.nn.functional as F
import json
import glob
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from utils.colmap_utils import load_colmap_calibration


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate point cloud from single depth map (batch-compatible)")
    parser.add_argument("-s", "--scene_dir", type=str, required=True,
                       help="Path to scene directory containing images/")
    parser.add_argument("-d", "--data_dir", type=str, required=True,
                       help="Path to VGGT batch-organized data directory")
    parser.add_argument("-i", "--idx", type=int, default=None,
                       help="Index of the image to generate point cloud for (0-based, optional)")
    parser.add_argument("-n", "--image_name", type=str, default=None,
                       help="Name of the image to process (alternative to --idx)")
    parser.add_argument("-b", "--batch_idx", type=int, default=None,
                       help="Specific batch index to use (optional, will auto-discover if not specified)")
    parser.add_argument("-c", "--conf_threshold", type=float, default=2.0,
                       help="Confidence threshold for filtering points (default: 2.0)")
    parser.add_argument("-o", "--output_dir", type=str, default="pointclouds",
                       help="Output directory for point cloud (default: scene_dir/pointclouds)")
    parser.add_argument("-r", "--vggt_model_resolution", type=int, default=518,
                       help="VGGT model resolution (default: 518)")
    parser.add_argument("--use_colmap", action="store_true", default=False,
                       help="Use COLMAP calibration instead of individual camera parameters")
    parser.add_argument("--list_batches", action="store_true", default=False,
                       help="List available batches and exit")
    parser.add_argument("--list_images", action="store_true", default=False,
                       help="List available images in batches and exit")
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


def find_available_batches(data_dir):
    """Find all available batch directories."""
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    batch_dirs = []
    for item in sorted(os.listdir(data_dir)):
        if item.startswith('batch_') and os.path.isdir(os.path.join(data_dir, item)):
            batch_dirs.append(item)
    
    return batch_dirs


def load_batch_metadata(batch_dir):
    """Load metadata for a batch."""
    metadata_file = os.path.join(batch_dir, 'batch_metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return None


def find_image_in_batches(data_dir, image_name):
    """Find which batch contains the specified image."""
    batch_dirs = find_available_batches(data_dir)
    
    for batch_name in batch_dirs:
        batch_path = os.path.join(data_dir, batch_name)
        
        # Check metadata first
        metadata = load_batch_metadata(batch_path)
        if metadata and 'image_names' in metadata:
            if image_name in metadata['image_names']:
                return batch_name, batch_path
        
        # Fallback: check if files exist
        raw_data_dir = os.path.join(batch_path, 'raw_data')
        if os.path.exists(raw_data_dir):
            base_name = os.path.splitext(image_name)[0]
            depth_file = os.path.join(raw_data_dir, f"{base_name}_depth.npy")
            if os.path.exists(depth_file):
                return batch_name, batch_path
    
    return None, None


def list_batches_and_images(data_dir):
    """List all available batches and their images."""
    batch_dirs = find_available_batches(data_dir)
    
    if not batch_dirs:
        print(f"❌ No batch directories found in {data_dir}")
        return
    
    print(f"📁 Found {len(batch_dirs)} batches in {data_dir}:")
    print("=" * 60)
    
    for batch_name in batch_dirs:
        batch_path = os.path.join(data_dir, batch_name)
        metadata = load_batch_metadata(batch_path)
        
        print(f"\n🗂️  {batch_name}:")
        if metadata:
            print(f"   📊 Images: {metadata.get('total_images_in_batch', 'unknown')}")
            image_names = metadata.get('image_names', [])
            if image_names:
                print(f"   🖼️  Image names: {', '.join(image_names[:5])}")
                if len(image_names) > 5:
                    print(f"       ... and {len(image_names) - 5} more")
        else:
            # Fallback: check raw_data directory
            raw_data_dir = os.path.join(batch_path, 'raw_data')
            if os.path.exists(raw_data_dir):
                depth_files = [f for f in os.listdir(raw_data_dir) if f.endswith('_depth.npy')]
                image_names = [f[:-10] + '.jpg' for f in depth_files[:5]]  # Assume .jpg extension
                print(f"   📊 Estimated images: {len(depth_files)}")
                print(f"   🖼️  Sample names: {', '.join(image_names)}")


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
    
    Args:
        colmap_intrinsic: 3x3 intrinsic matrix from COLMAP
        depth_width: Width of the depth map in pixels (e.g., 518)
        depth_height: Height of the depth map in pixels (e.g., 518) 
        original_image_path: Path to original image to get its resolution
    
    Returns:
        3x3 intrinsic matrix in VGGT format for depth map resolution
    """
    fx_colmap, fy_colmap = colmap_intrinsic[0, 0], colmap_intrinsic[1, 1]
    cx_colmap, cy_colmap = colmap_intrinsic[0, 2], colmap_intrinsic[1, 2]
    
    print(f"📷 COLMAP intrinsics: fx={fx_colmap:.6f}, fy={fy_colmap:.6f}, cx={cx_colmap:.6f}, cy={cy_colmap:.6f}")
    
    # Get original image resolution
    if original_image_path and os.path.exists(original_image_path):
        with Image.open(original_image_path) as img:
            orig_width, orig_height = img.size
        print(f"📏 Original image size: {orig_width}x{orig_height}")
    else:
        # Fallback to common resolution
        orig_width, orig_height = 4056, 3040
        print(f"⚠️  Original image not found, assuming {orig_width}x{orig_height}")
        
    # Check if intrinsics are normalized (typical signs: fx/fy around 1.0, cx/cy around 0.0)
    if abs(fx_colmap - 1.0) < 0.1 and abs(fy_colmap - 1.0) < 0.1:
        print("⚠️  Detected normalized intrinsics, converting to pixel space...")
        
        # Calculate scale factors from original image to depth map
        scale_x = depth_width / orig_width
        scale_y = depth_height / orig_height
        
        print(f"📐 Scale factors: x={scale_x:.6f}, y={scale_y:.6f}")
        
        # Estimate reasonable focal lengths for the original image
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
        # Scale from original image resolution to depth map resolution
        scale_x = depth_width / orig_width
        scale_y = depth_height / orig_height
        
        fx_vggt = fx_colmap * scale_x
        fy_vggt = fy_colmap * scale_y
        cx_vggt = cx_colmap * scale_x
        cy_vggt = cy_colmap * scale_y
        
        print(f"✅ Using COLMAP intrinsics scaled to depth map resolution: fx={fx_vggt:.6f}, fy={fy_vggt:.6f}, cx={cx_vggt:.6f}, cy={cy_vggt:.6f}")

    # Construct VGGT-format intrinsic matrix
    vggt_intrinsic = np.array([
        [fx_vggt, 0, cx_vggt],
        [0, fy_vggt, cy_vggt],
        [0, 0, 1]
    ])

    return vggt_intrinsic


def generate_single_pointcloud(scene_dir, data_dir, idx=None, image_name=None, batch_idx=None, 
                             conf_threshold=2.0, vggt_model_resolution=518, output_dir="pointclouds", 
                             use_colmap=False):
    """
    Generate point cloud for a single depth map from batch-organized data.
    
    Args:
        scene_dir: Path to scene directory containing images/
        data_dir: Path to VGGT batch-organized data directory
        idx: Index of the image to process (0-based, optional)
        image_name: Name of the image to process (alternative to idx)
        batch_idx: Specific batch index to use (optional)
        conf_threshold: Confidence threshold for filtering points
        vggt_model_resolution: VGGT model resolution
        output_dir: Output directory for point cloud
        use_colmap: Whether to use COLMAP calibration
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Setup paths
    images_dir = os.path.join(scene_dir, "images")
    output_dir = os.path.join(scene_dir, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔍 Scene directory: {scene_dir}")
    print(f"🗂️  Data directory: {data_dir}")
    print(f"📁 Images directory: {images_dir}")
    print(f"💾 Output directory: {output_dir}")
    
    try:
        # Determine target image
        if image_name is None:
            if idx is None:
                raise ValueError("Either idx or image_name must be specified")
            
            # Find image files and get by index
            image_files = find_image_files(images_dir)
            if idx >= len(image_files):
                raise ValueError(f"Image index {idx} is out of range. Found {len(image_files)} images.")
            
            target_image_name = image_files[idx]
            print(f"🎯 Target image (by index {idx}): {target_image_name}")
        else:
            target_image_name = image_name
            print(f"🎯 Target image (by name): {target_image_name}")
        
        # Find the batch containing this image
        if batch_idx is not None:
            batch_name = f"batch_{batch_idx:03d}"
            batch_path = os.path.join(data_dir, batch_name)
            if not os.path.exists(batch_path):
                raise ValueError(f"Specified batch directory not found: {batch_path}")
            print(f"📦 Using specified batch: {batch_name}")
        else:
            batch_name, batch_path = find_image_in_batches(data_dir, target_image_name)
            if batch_path is None:
                raise ValueError(f"Image {target_image_name} not found in any batch")
            print(f"🔍 Found image in batch: {batch_name}")
        
        # Setup batch-specific paths
        raw_data_dir = os.path.join(batch_path, "raw_data")
        individual_cameras_dir = os.path.join(batch_path, "individual_cameras")
        colmap_calibration_dir = os.path.join(batch_path, "colmap_calibration")
        
        print(f"🗂️  Raw data directory: {raw_data_dir}")
        if use_colmap:
            print(f"🎯 COLMAP calibration directory: {colmap_calibration_dir}")
        else:
            print(f"🎯 Individual cameras directory: {individual_cameras_dir}")
        
        # Find depth and confidence files
        depth_files, confidence_files = find_depth_confidence_files(raw_data_dir)
        
        # Create mapping from base names to files
        depth_map = {base: filename for base, filename in depth_files}
        confidence_map = {base: filename for base, filename in confidence_files}
        
        # Find matching depth and confidence files for the target image
        base_name = os.path.splitext(target_image_name)[0]
        
        if base_name not in depth_map:
            raise ValueError(f"No depth map found for image {target_image_name} in {batch_name}")
        if base_name not in confidence_map:
            raise ValueError(f"No confidence map found for image {target_image_name} in {batch_name}")
        
        depth_file = os.path.join(raw_data_dir, depth_map[base_name])
        confidence_file = os.path.join(raw_data_dir, confidence_map[base_name])
        
        print(f"🗺️  Depth map: {depth_file}")
        print(f"📊 Confidence map: {confidence_file}")
        
        # Load camera parameters (either individual or from COLMAP)
        if use_colmap:
            # Load from COLMAP calibration
            calibration_data = load_colmap_calibration(colmap_calibration_dir)
            
            if calibration_data is None:
                print(f"❌ Failed to load COLMAP calibration from {colmap_calibration_dir}")
                return False
            
            print(f"✅ Loaded COLMAP calibration from {colmap_calibration_dir}")
            
            # Get extrinsic from COLMAP
            if target_image_name in calibration_data['images']:
                extrinsic = calibration_data['images'][target_image_name]['extrinsic']
                print(f"✅ Using COLMAP extrinsics for {target_image_name}")
            else:
                print(f"❌ Target image {target_image_name} not found in COLMAP calibration")
                # Print available images for debugging
                available_images = list(calibration_data['images'].keys())[:5]  # Show first 5
                print(f"📋 Available images in COLMAP (first 5): {available_images}")
                return False
            
            # Load individual intrinsics from batch individual_cameras directory
            individual_intrinsics_path = os.path.join(individual_cameras_dir, f"{base_name}_intrinsic.npy")
            
            if os.path.exists(individual_intrinsics_path):
                intrinsic = np.load(individual_intrinsics_path)
                print(f"✅ Loaded individual intrinsics from {individual_intrinsics_path}")
                print(f"📷 Individual intrinsics: fx={intrinsic[0,0]:.2f}, fy={intrinsic[1,1]:.2f}, cx={intrinsic[0,2]:.2f}, cy={intrinsic[1,2]:.2f}")
            else:
                print(f"❌ Individual intrinsics not found at {individual_intrinsics_path}")
                print(f"🔄 Falling back to COLMAP intrinsics...")
                
                # Fallback to COLMAP intrinsics with conversion
                colmap_intrinsic = calibration_data['images'][target_image_name]['intrinsic']
                intrinsic = convert_colmap_intrinsics_to_vggt_format(
                    colmap_intrinsic, vggt_model_resolution, vggt_model_resolution,
                    original_image_path=os.path.join(scene_dir, "images", target_image_name)
                )
        else:
            # Load individual camera parameters
            from utils.colmap_utils import load_individual_camera_parameters
            camera_data = load_individual_camera_parameters(base_name, individual_cameras_dir)
            if camera_data is None:
                raise ValueError(f"No camera parameters found for image {target_image_name} in {individual_cameras_dir}")
            
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
        output_path = os.path.join(output_dir, f"{base_name}_pointcloud_{batch_name}.ply")
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
    
    print("🌤️  Single Depth Map Point Cloud Generator (Batch-Compatible)")
    print("=" * 70)
    
    # Handle listing commands
    if args.list_batches or args.list_images:
        list_batches_and_images(args.data_dir)
        return
    
    # Validate arguments
    if args.idx is None and args.image_name is None:
        print("❌ Error: Either --idx or --image_name must be specified")
        print("💡 Use --list_images to see available images")
        return
    
    success = generate_single_pointcloud(
        scene_dir=args.scene_dir,
        data_dir=args.data_dir,
        idx=args.idx,
        image_name=args.image_name,
        batch_idx=args.batch_idx,
        conf_threshold=args.conf_threshold,
        vggt_model_resolution=args.vggt_model_resolution,
        output_dir=args.output_dir,
        use_colmap=args.use_colmap
    )
    
    if success:
        print("\n✅ Point cloud generation completed successfully!")
    else:
        print("\n❌ Point cloud generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main() 