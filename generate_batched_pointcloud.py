# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT Batch Point Cloud Generation

Generate point clouds from image sequences using VGGT model with batched processing.
Each batch creates its own directory containing all related assets (point clouds, 
depth maps, camera calibration, etc.) for easy organization and processing.

Batching Strategies:
    1. Neighbor-based (default): Groups images that share the most 3D points in the 
       reference calibration, creating batches with better visual overlap
    2. Sequential: Traditional approach processing images in order

Point Cloud Computation:
    - Uses VGGT intrinsics with reference calibration extrinsics for unprojection
    - Saves VGGT extrinsics to vggt_calibration/ for scale estimation against reference
    - Estimates scale factor between VGGT and reference coordinate systems
    - Scales VGGT depth maps with estimated scale before unprojection 
    - Computes point clouds with scaled depth maps and reference extrinsics
    - Saves reference extrinsics to transformed/ as final output
    - Results are directly in reference coordinate system with correct scale

Output Structure:
    output_dir/
    ├── pointcloud.ply               # Global combined point cloud (all batches)
    ├── batch_000/
    │   ├── ply/                     # Point clouds (.ply files)
    │   │   ├── combined.ply         # All batch images combined
    │   │   ├── image1.ply           # Individual image point clouds
    │   │   └── image2.ply
    │   ├── depth/                   # Colorized depth maps (.png)
    │   ├── raw_data/                # Raw depth & confidence (.npy)
    │   ├── individual_cameras/      # Camera parameters (.npy)
    │   ├── vggt_calibration/      # VGGT calibration (for scale estimation)
    │   ├── transformed/             # Reference calibration & scale results
    │   │   ├── scale.json           # Scale estimation info
    │   │   ├── cameras.txt          # Reference COLMAP calibration (final output)
    │   │   ├── images.txt
    │   │   ├── points3D.txt
    │   │   └── ply/                 # Transformed point clouds
    │   │       ├── combined.ply     # Transformed combined point cloud
    │   │       ├── image1.ply       # Transformed individual point clouds
    │   │       └── image2.ply
    │   └── batch_metadata.json     # Batch processing info
    ├── batch_001/
    │   └── ...
    └── processing_metadata.json    # Overall processing info

Examples:
    # Basic usage with neighbor-based batching (default)
    python3 generate_batched_pointcloud.py -s scene/ -o output/ -g reference_colmap/

    # Specify batch size and resolution with neighbor-based batching
    python3 generate_batched_pointcloud.py -s scene/ -o results/ -g reference_colmap/ -b 16 -r 512

    # Force sequential batching instead of neighbor-based
    python3 generate_batched_pointcloud.py -s scene/ -o output/ -g reference_colmap/ --sequential_batching

    # Use absolute output path with neighbor-based batching
    python3 generate_batched_pointcloud.py -s scene/ -o /tmp/pointclouds/ -g reference_colmap/

    # Limit number of images and set confidence threshold
    python3 generate_batched_pointcloud.py -s scene/ -o output/ -g reference_colmap/ -m 50 -c 1.5

    # Process with custom settings
    python3 generate_batched_pointcloud.py -s scene/ -o output/ -g reference_colmap/ -b 4 -c 2.5 --colormap jet
"""

import random
import numpy as np
import glob
import json
import os
import torch
import torch.nn.functional as F
import gc
import trimesh
import matplotlib.cm as cm
from PIL import Image

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from utils.colmap_utils import save_vggt_calibration_as_colmap, save_individual_camera_parameters

from utils.reconstruction_transform import (
    estimate_similarity_transform_from_recons,
    load_reconstruction,
    apply_similarity_transform_to_reconstruction,
    save_reconstruction_text,
    transform_point_cloud_to_colmap_frame,
    extract_camera_centers_and_rotations,
    apply_similarity_transform_to_point,
)
from collections import defaultdict, Counter

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Batch Point Estimation")
    parser.add_argument("-s", "--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save the output point clouds and depth maps (relative to scene_dir if not absolute)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("-r", "--resolution", type=int, default=518, help="Preprocessing resolution. Model always runs at 518.")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Number of images to process together.")
    parser.add_argument("-m", "--max_images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("-c", "--conf_threshold", type=float, default=2.0, help="Confidence threshold to filter points (from depth head, >1).")
    parser.add_argument("--colormap", type=str, default="viridis", help="Colormap for depth visualization (e.g., viridis, jet, inferno).")
    parser.add_argument("-g", "--reference_calibration", type=str, required=True, help="Directory containing a reference calibration in colmap format")
    parser.add_argument("--use_neighbor_batching", action="store_true", default=True, help="Use neighbor-based batching based on 3D point sharing (default: True)")
    parser.add_argument("--sequential_batching", action="store_true", default=False, help="Force sequential batching instead of neighbor-based (overrides --use_neighbor_batching)")

    parser.add_argument("--save_raw_data", action="store_true", default=True, help="Save raw depth and confidence maps as numpy arrays for later use")
    
    return parser.parse_args()


def colorize_depth_map(depth, cmap='viridis', min_percentile=5, max_percentile=95):
    """
    Colorizes a depth map for visualization using a specified colormap.
    """
    # Clamp depth values to a robust range
    valid_mask = np.isfinite(depth)
    if not valid_mask.any():
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    min_val = np.percentile(depth[valid_mask], min_percentile)
    max_val = np.percentile(depth[valid_mask], max_percentile)

    # Handle cases where min and max are the same
    if max_val <= min_val:
        normalized_depth = np.zeros_like(depth)
    else:
        normalized_depth = (depth - min_val) / (max_val - min_val)
        normalized_depth = np.clip(normalized_depth, 0, 1)

    # Use a colormap to colorize the depth map
    colored_depth = getattr(cm, cmap)(normalized_depth)[:, :, :3]
    colored_depth_uint8 = (colored_depth * 255).astype(np.uint8)

    return colored_depth_uint8


def run_VGGT_batch_pointcloud(model, images_batch, dtype, vggt_model_resolution=518):
    """
    Run VGGT for a batch of images to get depth maps and camera parameters.
    Note: Point clouds are computed separately using reference extrinsics.
    
    Args:
        model: VGGT model with camera and depth heads enabled
        images_batch: [B, 3, H, W] batch of images at model resolution
        dtype: Data type for mixed precision
        vggt_model_resolution: Fixed resolution for VGGT model (518)
    
    Returns:
        points_3d: Numpy array of 3D points [B, H, W, 3] (computed with VGGT extrinsics, will be replaced)
        depth_conf: Numpy array of depth confidence [B, H, W]
        depth_map: Numpy array of depth maps [B, H, W, 1]
        images_for_color: Torch tensor of images for coloring points [B, 3, H, W]
        extrinsic: VGGT Camera extrinsic matrices [B, 3, 4] (not used for final point clouds)
        intrinsic: VGGT Camera intrinsic matrices [B, 3, 3] (used for final point clouds)
    """
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            # Expected input shape for aggregator: [1, B, 3, H, W]
            vggt_input = images_batch.unsqueeze(0)
            
            aggregated_tokens_list, ps_idx = model.aggregator(vggt_input)
            
            # 1. Predict Cameras (for unprojection)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, vggt_input.shape[-2:])
            
            # 2. Predict Depth
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images=vggt_input, patch_start_idx=ps_idx)

    # Squeeze the outer batch dimension (B=1) that was added for the model
    extrinsic_np = extrinsic.squeeze(0).cpu().numpy()  # Shape: [S, 3, 4]
    intrinsic_np = intrinsic.squeeze(0).cpu().numpy()  # Shape: [S, 3, 3]
    depth_map_np = depth_map.squeeze(-1).squeeze(0).cpu().numpy()   # Shape: [S, H, W]
    depth_conf_np = depth_conf.squeeze(0).cpu().numpy() # Shape: [S, H, W]

    # 3. Unproject depth to get 3D points
    # Add the channel dimension back for unprojection
    depth_map_for_unproject = depth_map_np[..., None]  # [S, H, W, 1]
    points_3d = unproject_depth_map_to_point_map(depth_map_for_unproject, extrinsic_np, intrinsic_np)
    
    return points_3d, depth_conf_np, depth_map_np, images_batch.cpu(), extrinsic_np, intrinsic_np

def estimate_scale_only(source_sparse_dir, target_sparse_dir, out_dir):
    """
    Estimate only the scale component from source to target reconstruction.
    
    Args:
        source_sparse_dir: Path to source COLMAP reconstruction  
        target_sparse_dir: Path to target COLMAP reconstruction
        out_dir: Output directory for scale info
    
    Returns:
        float: Estimated scale factor
    """
    result = estimate_similarity_transform_from_recons(
        source_sparse_dir=source_sparse_dir,
        target_sparse_dir=target_sparse_dir,
        robust_scale=True,
    )

    print("=== Scale Estimation (source -> target) ===")
    print(f"Common images: {result['num_common']}")
    if result['num_common'] <= 10:
        print(f"Names: {result['common_images']}")
    print(f"Scale: {result['scale']:.9f}")
    print(f"RMSE (centers): {result['rmse']:.9f}")

    # Save only scale information
    os.makedirs(out_dir, exist_ok=True)
    scale_json_out = os.path.join(out_dir, "scale.json")
    with open(scale_json_out, "w") as f:
        json.dump({
            "scale": float(result['scale']),
            "rmse": float(result['rmse']),
            "num_common": int(result['num_common']),
            "common_images": result['common_images'],
        }, f, indent=2)
    print(f"Wrote scale info to {scale_json_out}")

    return result['scale']


def analyze_3d_point_sharing(reference_calibration_dir):
    """
    Analyze 3D point sharing between images in the reference calibration.
    
    Args:
        reference_calibration_dir: Path to reference COLMAP reconstruction
    
    Returns:
        dict: {
            'image_to_points': dict mapping image names to sets of 3D point IDs,
            'point_to_images': dict mapping 3D point IDs to sets of image names,
            'image_names': list of all image names in reconstruction
        }
    """
    try:
        # Load reference reconstruction
        reconstruction = load_reconstruction(reference_calibration_dir)
        
        # Initialize data structures
        image_to_points = defaultdict(set)
        point_to_images = defaultdict(set)
        image_names = []
        
        # Get all registered image names
        for image_id, image in reconstruction.images.items():
            if image.registered:
                image_names.append(image.name)
        
        # Analyze 3D point tracks
        for point3d_id, point3d in reconstruction.points3D.items():
            track = point3d.track
            
            # Each track element contains image_id and point2D_idx
            for track_element in track.elements:
                image_id = track_element.image_id
                
                # Get image name from image_id
                if image_id in reconstruction.images:
                    image = reconstruction.images[image_id]
                    if image.registered:
                        image_name = image.name
                        
                        # Record the association
                        image_to_points[image_name].add(point3d_id)
                        point_to_images[point3d_id].add(image_name)
        
        print(f"📊 3D Point Analysis:")
        print(f"  Total registered images: {len(image_names)}")
        print(f"  Total 3D points: {len(reconstruction.points3D)}")
        
        # Convert defaultdict to regular dict for cleaner output
        return {
            'image_to_points': dict(image_to_points),
            'point_to_images': dict(point_to_images),
            'image_names': sorted(image_names)
        }
        
    except Exception as e:
        print(f"❌ Error analyzing 3D point sharing: {e}")
        return None


def find_best_neighbors(target_image, point_sharing_info, batch_size, excluded_images=None):
    """
    Find the best neighboring images that share the most 3D points with the target image.
    
    Args:
        target_image: Name of the target image
        point_sharing_info: Output from analyze_3d_point_sharing()
        batch_size: Number of images to include in batch (including target)
        excluded_images: Set of image names to exclude (already processed)
    
    Returns:
        list: List of image names for the batch (including target image)
    """
    if excluded_images is None:
        excluded_images = set()
    
    image_to_points = point_sharing_info['image_to_points']
    
    if target_image not in image_to_points:
        print(f"⚠️  Warning: Target image {target_image} not found in reference calibration")
        return [target_image]
    
    target_points = image_to_points[target_image]
    
    # Calculate shared points with all other images
    shared_counts = Counter()
    
    for other_image, other_points in image_to_points.items():
        if other_image != target_image and other_image not in excluded_images:
            shared_points = len(target_points.intersection(other_points))
            if shared_points > 0:
                shared_counts[other_image] = shared_points
    
    # Get the top (batch_size - 1) neighbors
    best_neighbors = [img for img, count in shared_counts.most_common(batch_size - 1)]
    
    # Create the batch with target image first
    batch = [target_image] + best_neighbors
    
    # Report sharing statistics
    print(f"  🎯 Target: {target_image} (has {len(target_points)} 3D points)")
    for neighbor in best_neighbors:
        if neighbor in image_to_points:
            neighbor_points = image_to_points[neighbor]
            shared = len(target_points.intersection(neighbor_points))
            print(f"    📌 {neighbor}: {shared} shared points (has {len(neighbor_points)} total)")
    
    return batch


def create_neighbor_based_batches(image_paths, point_sharing_info, batch_size):
    """
    Create batches based on 3D point sharing rather than sequential ordering.
    
    Args:
        image_paths: List of all image paths to process
        point_sharing_info: Output from analyze_3d_point_sharing()
        batch_size: Size of each batch
    
    Returns:
        list: List of batches, where each batch is a list of image paths
    """
    # Extract image names from paths for matching with reference calibration
    path_to_name = {path: os.path.basename(path) for path in image_paths}
    name_to_path = {os.path.basename(path): path for path in image_paths}
    
    # Get available images that are in both our dataset and reference calibration
    available_images = []
    for path in image_paths:
        image_name = os.path.basename(path)
        if image_name in point_sharing_info['image_to_points']:
            available_images.append(image_name)
        else:
            print(f"⚠️  Warning: {image_name} not found in reference calibration, skipping")
    
    print(f"📋 Neighbor-based batching:")
    print(f"  Total images to process: {len(image_paths)}")
    print(f"  Images available in reference: {len(available_images)}")
    print(f"  Batch size: {batch_size}")
    
    batches = []
    processed_images = set()
    
    # Process images in order, but form batches based on 3D point sharing
    for i, image_name in enumerate(available_images):
        if image_name in processed_images:
            continue
        
        print(f"\n🗂️  Creating batch {len(batches) + 1} with target {image_name}:")
        
        # Find best neighbors for this image
        batch_image_names = find_best_neighbors(
            image_name, 
            point_sharing_info, 
            batch_size, 
            excluded_images=processed_images
        )
        
        # Convert image names back to paths
        batch_paths = []
        for name in batch_image_names:
            if name in name_to_path and name not in processed_images:
                batch_paths.append(name_to_path[name])
                processed_images.add(name)
        
        if batch_paths:
            batches.append(batch_paths)
            print(f"    ✅ Batch {len(batches)}: {len(batch_paths)} images")
        
        # Stop if we've processed all images
        if len(processed_images) >= len(available_images):
            break
    
    print(f"\n📊 Batching Summary:")
    print(f"  Total batches created: {len(batches)}")
    print(f"  Images processed: {len(processed_images)}")
    print(f"  Images skipped: {len(image_paths) - len(processed_images)}")
    
    return batches


def load_reference_extrinsics_for_batch(batch_image_names, reference_calibration_dir):
    """
    Load reference calibration extrinsics for the given batch images.
    
    Args:
        batch_image_names: List of image names in the batch
        reference_calibration_dir: Path to reference COLMAP reconstruction
    
    Returns:
        np.ndarray: Array of extrinsic matrices [B, 3, 4]
    """
    try:
        # Load reference reconstruction
        from utils.colmap_utils import load_colmap_calibration
        reference_data = load_colmap_calibration(reference_calibration_dir)
        
        if reference_data is None:
            raise ValueError(f"Failed to load reference calibration from {reference_calibration_dir}")
        
        reference_images = reference_data['images']
        extrinsics_list = []
        
        for image_name in batch_image_names:
            if image_name in reference_images:
                extrinsic = reference_images[image_name]['extrinsic']
                extrinsics_list.append(extrinsic)
                print(f"    ✅ {image_name}: Found reference extrinsic")
            else:
                raise ValueError(f"Image {image_name} not found in reference calibration")
        
        # Stack into batch format [B, 3, 4]
        reference_extrinsics = np.stack(extrinsics_list, axis=0)
        
        print(f"  ✅ Loaded reference extrinsics for {len(batch_image_names)} images")
        return reference_extrinsics
        
    except Exception as e:
        print(f"  ❌ Error loading reference extrinsics: {e}")
        raise


def recompute_pointclouds_with_reference_extrinsics(depth_maps, intrinsics, reference_extrinsics):
    """
    Recompute 3D point clouds using VGGT depth maps and intrinsics with reference extrinsics.
    
    Args:
        depth_maps: VGGT depth maps [B, H, W]
        intrinsics: VGGT intrinsic matrices [B, 3, 3]
        reference_extrinsics: Reference extrinsic matrices [B, 3, 4]
    
    Returns:
        np.ndarray: 3D points [B, H, W, 3]
    """
    try:
        # Add channel dimension for unprojection: [B, H, W] -> [B, H, W, 1]
        depth_maps_for_unproject = depth_maps[..., None]
        
        # Unproject using reference extrinsics and VGGT intrinsics
        points_3d = unproject_depth_map_to_point_map(
            depth_maps_for_unproject, reference_extrinsics, intrinsics
        )
        
        print(f"  ✅ Recomputed point clouds using reference extrinsics")
        print(f"     Shape: {points_3d.shape}")
        
        return points_3d
        
    except Exception as e:
        print(f"  ❌ Error recomputing point clouds: {e}")
        raise


def process_images_for_pointclouds(model, image_paths, dtype, args):
    """
    Process images in batches to generate and save point clouds and depth maps.
    Each batch gets its own directory with all related assets.
    """
    vggt_model_resolution = 518
    
    # Limit number of images if specified
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]
    
    # Determine batching strategy
    use_neighbor_batching = args.use_neighbor_batching and not args.sequential_batching
    
    if use_neighbor_batching:
        print(f"🔍 Analyzing 3D point sharing in reference calibration...")
        # Analyze 3D point sharing from reference calibration
        point_sharing_info = analyze_3d_point_sharing(args.reference_calibration)
        
        if point_sharing_info is None:
            print("❌ Failed to analyze 3D point sharing, falling back to sequential batching")
            use_neighbor_batching = False
        else:
            print(f"✅ Using neighbor-based batching based on 3D point sharing")
            batches = create_neighbor_based_batches(image_paths, point_sharing_info, args.batch_size)
    
    if not use_neighbor_batching:
        print(f"📋 Using sequential batching")
        # Sequential batching
        num_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
        batches = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(image_paths))
            batches.append(image_paths[start_idx:end_idx])
    
    num_batches = len(batches)
    print(f"\n🚀 Processing {len(image_paths)} images in {num_batches} batches")
    print(f"Preprocessing at {args.resolution}x{args.resolution}, model runs at {vggt_model_resolution}x{vggt_model_resolution}")
    
    for batch_idx, batch_paths in enumerate(batches):
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}: {len(batch_paths)} images")
        print(f"  Images: {[os.path.basename(p) for p in batch_paths]}")
        
        # Create batch-specific directory structure
        batch_dir = os.path.join(args.output_dir, f"batch_{batch_idx:03d}")
        ply_output_dir = os.path.join(batch_dir, "ply")
        depth_output_dir = os.path.join(batch_dir, "depth")
        raw_data_dir = os.path.join(batch_dir, "raw_data")
        vggt_calibration_dir = os.path.join(batch_dir, "vggt_calibration")
        individual_cameras_dir = os.path.join(batch_dir, "individual_cameras")
        transformed_dir = os.path.join(batch_dir, "transformed")
        
        os.makedirs(ply_output_dir, exist_ok=True)
        os.makedirs(depth_output_dir, exist_ok=True)
        if args.save_raw_data:
            os.makedirs(raw_data_dir, exist_ok=True)
            os.makedirs(individual_cameras_dir, exist_ok=True)
        os.makedirs(vggt_calibration_dir, exist_ok=True)
        
        # Load batch with aspect-ratio preservation
        images, _ = load_and_preprocess_images_square(batch_paths, args.resolution)
        
        # Resize to model resolution
        images_for_model = F.interpolate(images, size=(vggt_model_resolution, vggt_model_resolution), mode="bilinear", align_corners=False)
        images_for_model = images_for_model.to(next(model.parameters()).device)

        # Get batch image names for reference loading and similarity transform
        batch_image_names = [os.path.basename(p) for p in batch_paths]
        
        # Process batch to get depth maps and VGGT intrinsics
        points_3d_batch, depth_conf_batch, depth_map_batch, images_for_color, vggt_extrinsic_batch, intrinsic_batch = run_VGGT_batch_pointcloud(
            model, images_for_model, dtype, vggt_model_resolution
        )
        
        # Load reference calibration extrinsics for the batch images
        print(f"  🔄 Loading reference extrinsics for batch images...")
        reference_extrinsics = load_reference_extrinsics_for_batch(batch_image_names, args.reference_calibration)
        
        # Save raw data if requested
        if args.save_raw_data:
            for i in range(len(batch_paths)):
                base_name = os.path.splitext(batch_image_names[i])[0]
                
                # Save raw depth and confidence maps
                depth_file = os.path.join(raw_data_dir, f"{base_name}_depth.npy")
                conf_file = os.path.join(raw_data_dir, f"{base_name}_confidence.npy")
                
                np.save(depth_file, depth_map_batch[i])
                np.save(conf_file, depth_conf_batch[i])
                print(f"  Saved raw depth map to {depth_file}")
                print(f"  Saved confidence map to {conf_file}")
        
        # Save COLMAP format calibration (using VGGT extrinsics + intrinsics for scale estimation)
        save_vggt_calibration_as_colmap(
            [vggt_extrinsic_batch], [intrinsic_batch], [batch_image_names], 
            vggt_calibration_dir, vggt_model_resolution
        )
        
        # Save individual camera parameters (for generate_cloud.py) - using VGGT extrinsics
        if args.save_raw_data:
            save_individual_camera_parameters(vggt_extrinsic_batch, intrinsic_batch, batch_image_names, individual_cameras_dir)
        

        # Estimate scale from the batch to the reference calibration
        estimated_scale = estimate_scale_only(vggt_calibration_dir, args.reference_calibration, transformed_dir)
        
        print(f"  📐 Estimated scale: {estimated_scale:.6f}")
        print(f"  🔄 Scaling depth maps before point cloud computation...")
        
        # Scale the depth maps with the estimated scale
        scaled_depth_maps = depth_map_batch * estimated_scale
        
        # Recompute point clouds using scaled depth maps with reference extrinsics  
        print(f"  🔄 Recomputing point clouds with scaled depth maps...")
        points_3d_batch = recompute_pointclouds_with_reference_extrinsics(
            scaled_depth_maps, intrinsic_batch, reference_extrinsics
        )
        
        print(f"  ✅ Point clouds computed with correct scale and reference poses")
        
        # Create transformed point clouds directory
        transformed_ply_dir = os.path.join(transformed_dir, "ply")
        os.makedirs(transformed_ply_dir, exist_ok=True)

        # Save reference extrinsics to transformed directory as the final output
        print(f"  💾 Saving reference extrinsics to transformed directory...")
        save_vggt_calibration_as_colmap(
            [reference_extrinsics], [intrinsic_batch], [batch_image_names], 
            transformed_dir, vggt_model_resolution
        )
        print(f"  ✅ Saved reference calibration as final transformed output")

        # --- Save Combined Batch Point Cloud ---
        # Flatten all points and confidences from the batch
        batch_points_flat = points_3d_batch.reshape(-1, 3)
        batch_conf_flat = depth_conf_batch.flatten()
        
        # Load and process original images for accurate color sampling
        batch_colors_list = []
        images_dir = os.path.join(args.scene_dir, "images")
        
        for i, image_path in enumerate(batch_paths):
            if os.path.exists(image_path):
                # Load and resize original image to match depth map resolution
                image = Image.open(image_path).convert('RGB')
                depth_h, depth_w = depth_conf_batch[i].shape
                image_resized = image.resize((depth_w, depth_h), Image.Resampling.LANCZOS)
                image_array = np.array(image_resized)  # HxWx3, uint8 [0, 255]
                batch_colors_list.append(image_array)
            else:
                print(f"⚠️  Warning: Original image not found at {image_path}, using gray colors")
                # Fallback to gray colors
                depth_h, depth_w = depth_conf_batch[i].shape
                gray_image = np.full((depth_h, depth_w, 3), 128, dtype=np.uint8)
                batch_colors_list.append(gray_image)
        
        # Stack all color images and flatten
        batch_colors_np = np.stack(batch_colors_list, axis=0)  # [B, H, W, 3]
        batch_colors_flat = batch_colors_np.reshape(-1, 3)

        # Filter the combined points
        combined_conf_mask = batch_conf_flat > args.conf_threshold
        combined_filtered_points = batch_points_flat[combined_conf_mask]
        combined_filtered_colors = batch_colors_flat[combined_conf_mask]
        
        # Save the combined point cloud
        combined_ply_path = os.path.join(ply_output_dir, "combined.ply")
        combined_pc = trimesh.PointCloud(vertices=combined_filtered_points, colors=combined_filtered_colors)
        combined_pc.export(combined_ply_path)
        print(f"  Saved combined batch with {combined_filtered_points.shape[0]} points to {combined_ply_path}")
        
        # Save the combined point cloud to transformed directory (no additional transformation needed)
        if combined_filtered_points.shape[0] > 0:
            try:
                # No transformation needed since points are already correctly scaled and positioned
                transformed_combined_path = os.path.join(transformed_ply_dir, "combined.ply")
                transformed_combined_pc = trimesh.PointCloud(vertices=combined_filtered_points, colors=combined_filtered_colors)
                transformed_combined_pc.export(transformed_combined_path)
                print(f"  Saved final combined batch with {combined_filtered_points.shape[0]} points to {transformed_combined_path}")
                
            except Exception as e:
                print(f"  ⚠️  Warning: Failed to save final point cloud: {e}")
        else:
            print(f"  ⚠️  No points in combined batch, skipping save")
        
        # --- Save Individual Frame Outputs ---
        # Save one point cloud and one depth map per frame in the batch
        for i in range(len(batch_paths)):
            base_name = os.path.basename(batch_paths[i])
            file_name_no_ext = os.path.splitext(base_name)[0]

            # --- Save Point Cloud ---
            frame_points = points_3d_batch[i]       # HxWx3
            frame_conf = depth_conf_batch[i]         # HxW
            
            # Filter points based on confidence
            conf_mask = frame_conf > args.conf_threshold
            filtered_points = frame_points[conf_mask]
            
            # Load original image for accurate color sampling
            images_dir = os.path.join(args.scene_dir, "images")
            original_image_path = batch_paths[i]
            
            if os.path.exists(original_image_path):
                # Load and resize original image to match depth map resolution
                image = Image.open(original_image_path).convert('RGB')
                depth_h, depth_w = frame_conf.shape
                image_resized = image.resize((depth_w, depth_h), Image.Resampling.LANCZOS)
                image_array = np.array(image_resized)  # HxWx3, uint8 [0, 255]
                
                # Sample colors using the confidence mask
                filtered_colors = image_array[conf_mask]
            else:
                print(f"⚠️  Warning: Original image not found at {original_image_path}, using gray colors")
                # Fallback to gray colors
                filtered_colors = np.zeros((len(filtered_points), 3), dtype=np.uint8)
                filtered_colors[:, :] = 128
            
            # Save individual point cloud
            output_filename = os.path.join(ply_output_dir, f"{file_name_no_ext}.ply")
            point_cloud = trimesh.PointCloud(vertices=filtered_points, colors=filtered_colors)
            point_cloud.export(output_filename)
            print(f"  Saved {filtered_points.shape[0]} points to {output_filename}")
            
            # Save individual point cloud to transformed directory (no additional transformation needed)
            if filtered_points.shape[0] > 0:
                try:
                    # No transformation needed since points are already correctly scaled and positioned
                    transformed_output_filename = os.path.join(transformed_ply_dir, f"{file_name_no_ext}.ply")
                    transformed_point_cloud = trimesh.PointCloud(vertices=filtered_points, colors=filtered_colors)
                    transformed_point_cloud.export(transformed_output_filename)
                    print(f"  Saved final {filtered_points.shape[0]} points to {transformed_output_filename}")
                    
                except Exception as e:
                    print(f"  ⚠️  Warning: Failed to save final point cloud for {file_name_no_ext}: {e}")
            else:
                print(f"  No points passed confidence threshold for {file_name_no_ext}")
            
            # --- Save Depth Map ---
            frame_depth = depth_map_batch[i]
            colored_depth = colorize_depth_map(frame_depth, cmap=args.colormap)
            
            depth_output_filename = os.path.join(depth_output_dir, f"{file_name_no_ext}_depth.png")
            Image.fromarray(colored_depth).save(depth_output_filename)
            print(f"  Saved depth map to {depth_output_filename}")

        # Save batch-specific metadata
        save_batch_metadata(args, batch_paths, batch_image_names, batch_dir, batch_idx, use_neighbor_batching)

        # Aggressive memory cleanup
        del images, images_for_model, points_3d_batch, depth_conf_batch, depth_map_batch, images_for_color
        torch.cuda.empty_cache()
        gc.collect()
    
    # Combine all transformed point clouds into a single global point cloud
    print(f"\n🔗 Combining all transformed point clouds...")
    combine_all_transformed_pointclouds(args.output_dir, num_batches)
    
    # Save overall processing metadata
    save_processing_metadata(args, image_paths, args.output_dir)


def combine_all_transformed_pointclouds(output_dir, num_batches):
    """
    Combine all transformed/ply/combined.ply files from all batches into a single global point cloud.
    
    Args:
        output_dir: Main output directory containing batch subdirectories
        num_batches: Number of batches processed
    """
    try:
        all_vertices = []
        all_colors = []
        successful_batches = 0
        total_points = 0
        
        print(f"  🔍 Searching for transformed point clouds in {num_batches} batches...")
        
        # Collect all transformed combined point clouds
        for batch_idx in range(num_batches):
            batch_dir = os.path.join(output_dir, f"batch_{batch_idx:03d}")
            transformed_combined_path = os.path.join(batch_dir, "transformed", "ply", "combined.ply")
            
            if os.path.exists(transformed_combined_path):
                try:
                    # Load the point cloud
                    point_cloud = trimesh.load(transformed_combined_path)
                    
                    if hasattr(point_cloud, 'vertices') and len(point_cloud.vertices) > 0:
                        vertices = point_cloud.vertices
                        colors = point_cloud.colors if hasattr(point_cloud, 'colors') else None
                        
                        all_vertices.append(vertices)
                        if colors is not None:
                            all_colors.append(colors)
                        else:
                            # Create default gray colors if colors are missing
                            gray_colors = np.full((len(vertices), 3), 128, dtype=np.uint8)
                            all_colors.append(gray_colors)
                        
                        successful_batches += 1
                        total_points += len(vertices)
                        print(f"    ✅ batch_{batch_idx:03d}: {len(vertices)} points")
                    else:
                        print(f"    ⚠️  batch_{batch_idx:03d}: No vertices found")
                        
                except Exception as e:
                    print(f"    ❌ batch_{batch_idx:03d}: Failed to load - {e}")
            else:
                print(f"    ⚠️  batch_{batch_idx:03d}: Transformed point cloud not found")
        
        if successful_batches == 0:
            print(f"  ❌ No transformed point clouds found to combine")
            return False
        
        # Combine all vertices and colors
        print(f"  🔗 Combining {successful_batches} point clouds...")
        combined_vertices = np.vstack(all_vertices)
        combined_colors = np.vstack(all_colors)
        
        # Create and save the combined point cloud
        global_pointcloud_path = os.path.join(output_dir, "pointcloud.ply")
        combined_pointcloud = trimesh.PointCloud(vertices=combined_vertices, colors=combined_colors)
        combined_pointcloud.export(global_pointcloud_path)
        
        print(f"  ✅ Successfully combined {successful_batches}/{num_batches} batches")
        print(f"     Total points: {len(combined_vertices):,}")
        print(f"     Saved to: {global_pointcloud_path}")
        print(f"     Point range: X[{combined_vertices[:,0].min():.3f}, {combined_vertices[:,0].max():.3f}], "
              f"Y[{combined_vertices[:,1].min():.3f}, {combined_vertices[:,1].max():.3f}], "
              f"Z[{combined_vertices[:,2].min():.3f}, {combined_vertices[:,2].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error combining point clouds: {e}")
        return False


def save_batch_metadata(args, batch_paths, batch_image_names, batch_dir, batch_idx, use_neighbor_batching):
    """Save metadata for a single batch."""
    import json
    
    batch_metadata = {
        'batch_index': batch_idx,
        'image_names': batch_image_names,
        'image_paths': batch_paths,
        'total_images_in_batch': len(batch_paths),
        'batch_directory': batch_dir,
        'processing_parameters': {
            'scene_dir': args.scene_dir,
            'output_dir': args.output_dir,
            'seed': args.seed,
            'resolution': args.resolution,
            'batch_size': args.batch_size,
            'max_images': args.max_images,
            'conf_threshold': args.conf_threshold,
            'colormap': args.colormap,
            'save_raw_data': args.save_raw_data,
            'vggt_model_resolution': 518,
            'reference_calibration': args.reference_calibration,
            'use_neighbor_batching': args.use_neighbor_batching,
            'sequential_batching': args.sequential_batching,
            'actual_batching_used': 'neighbor-based' if use_neighbor_batching else 'sequential'
        },
        'file_structure': {
            'ply_dir': 'ply/',
            'depth_dir': 'depth/',
            'raw_data_dir': 'raw_data/' if args.save_raw_data else None,
            'individual_cameras_dir': 'individual_cameras/' if args.save_raw_data else None,
            'vggt_calibration_dir': 'vggt_calibration/',
            'transformed_dir': 'transformed/',
            'transformed_ply_dir': 'transformed/ply/'
        },
        'file_formats': {
            'global_point_cloud': 'pointcloud.ply (combined from all batches, transformed to reference frame)',
            'point_clouds': '.ply (trimesh format)',
            'combined_point_cloud': 'combined.ply (all batch images combined)',
            'transformed_point_clouds': '.ply (trimesh format, transformed to reference frame)',
            'transformed_combined': 'transformed/ply/combined.ply (transformed combined point cloud)',
            'depth_maps': '.png (colorized visualization)',
            'raw_depth': '.npy (numpy array, float32)',
            'confidence': '.npy (numpy array, float32)',
            'extrinsics': '.npy (numpy array, shape [3, 4])',
            'intrinsics': '.npy (numpy array, shape [3, 3])',
            'vggt_calibration': 'VGGT sparse reconstruction format (cameras.txt, images.txt, points3D.txt)',
            'scale_data': 'scale.json (scale estimation info)'
        }
    }
    
    metadata_file = os.path.join(batch_dir, 'batch_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(batch_metadata, f, indent=2)
    
    print(f"  Batch metadata saved to {metadata_file}")


def save_processing_metadata(args, image_paths, output_dir):
    """Save metadata about the processing parameters and file structure."""
    import json
    
    # Calculate batch information
    num_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
    
    metadata = {
        'processing_parameters': {
            'scene_dir': args.scene_dir,
            'output_dir': output_dir,
            'seed': args.seed,
            'resolution': args.resolution,
            'batch_size': args.batch_size,
            'max_images': args.max_images,
            'conf_threshold': args.conf_threshold,
            'colormap': args.colormap,
            'save_raw_data': args.save_raw_data,
            'vggt_model_resolution': 518,
            'reference_calibration': args.reference_calibration,
            'use_neighbor_batching': args.use_neighbor_batching,
            'sequential_batching': args.sequential_batching
        },
        'batch_organization': {
            'total_batches': num_batches,
            'batch_directory_pattern': 'batch_XXX/',
            'description': 'Each batch has its own directory containing all related assets'
        },
        'file_structure_per_batch': {
            'ply_dir': 'ply/',
            'depth_dir': 'depth/',
            'raw_data_dir': 'raw_data/' if args.save_raw_data else None,
            'individual_cameras_dir': 'individual_cameras/' if args.save_raw_data else None,
            'vggt_calibration_dir': 'vggt_calibration/',
            'transformed_dir': 'transformed/',
            'transformed_ply_dir': 'transformed/ply/',
            'batch_metadata': 'batch_metadata.json'
        },
        'file_formats': {
            'global_point_cloud': 'pointcloud.ply (combined from all batches, transformed to reference frame)',
            'point_clouds': '.ply (trimesh format)',
            'combined_point_cloud': 'combined.ply (all batch images combined)',
            'transformed_point_clouds': '.ply (trimesh format, transformed to reference frame)',
            'transformed_combined': 'transformed/ply/combined.ply (transformed combined point cloud)',
            'depth_maps': '.png (colorized visualization)',
            'raw_depth': '.npy (numpy array, float32)',
            'confidence': '.npy (numpy array, float32)',
            'extrinsics': '.npy (numpy array, shape [3, 4])',
            'intrinsics': '.npy (numpy array, shape [3, 3])',
            'vggt_calibration': 'VGGT sparse reconstruction format (cameras.txt, images.txt, points3D.txt)',
            'scale_data': 'scale.json (scale estimation info)'
        },
        'data_info': {
            'total_images_processed': len(image_paths),
            'total_batches': num_batches,
            'image_names': [os.path.basename(p) for p in image_paths]
        }
    }
    
    metadata_file = os.path.join(output_dir, 'processing_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Processing metadata saved to {metadata_file}")


def find_image_batch(data_dir, image_name):
    """
    Find which batch directory contains the specified image.
    
    Args:
        data_dir: Root VGGT output directory
        image_name: Name of the image (without extension)
    
    Returns:
        str or None: Path to the batch directory containing the image, or None if not found
    """
    import json
    
    # Look through batch directories
    batch_dirs = [d for d in os.listdir(data_dir) if d.startswith('batch_') and os.path.isdir(os.path.join(data_dir, d))]
    
    for batch_dir in sorted(batch_dirs):
        batch_path = os.path.join(data_dir, batch_dir)
        metadata_file = os.path.join(batch_path, 'batch_metadata.json')
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                image_names_in_batch = [os.path.splitext(name)[0] for name in metadata.get('image_names', [])]
                if image_name in image_names_in_batch:
                    return batch_path
        
        # Fallback: check if the image files exist in this batch
        ply_file = os.path.join(batch_path, "ply", f"{image_name}.ply")
        if os.path.exists(ply_file):
            return batch_path
    
    return None


def load_vggt_data(data_dir, image_name):
    """
    Load VGGT data for a specific image from the batch-organized directory structure.
    
    Args:
        data_dir: Root directory containing the VGGT output (with batch_XXX subdirs)
        image_name: Name of the image (without extension)
    
    Returns:
        dict: Dictionary containing loaded data
    """
    from utils.colmap_utils import load_individual_camera_parameters
    
    # Find which batch contains this image
    batch_dir = find_image_batch(data_dir, image_name)
    if batch_dir is None:
        print(f"Warning: Could not find image {image_name} in any batch")
        return {}
    
    data = {'batch_directory': batch_dir}
    
    # Load raw depth and confidence maps
    raw_data_dir = os.path.join(batch_dir, "raw_data")
    if os.path.exists(raw_data_dir):
        depth_file = os.path.join(raw_data_dir, f"{image_name}_depth.npy")
        conf_file = os.path.join(raw_data_dir, f"{image_name}_confidence.npy")
        
        if os.path.exists(depth_file):
            data['depth_map'] = np.load(depth_file)
        if os.path.exists(conf_file):
            data['confidence_map'] = np.load(conf_file)
    
    # Load camera parameters (try numpy arrays first, then COLMAP format)
    cameras_dir = os.path.join(batch_dir, "individual_cameras")
    if os.path.exists(cameras_dir):
        camera_data = load_individual_camera_parameters(image_name, cameras_dir)
        if camera_data:
            data.update(camera_data)
    
    # Load point cloud
    ply_dir = os.path.join(batch_dir, "ply")
    ply_file = os.path.join(ply_dir, f"{image_name}.ply")
    if os.path.exists(ply_file):
        data['point_cloud'] = trimesh.load(ply_file)
    
    # Load colorized depth map
    depth_dir = os.path.join(batch_dir, "depth")
    depth_file = os.path.join(depth_dir, f"{image_name}_depth.png")
    if os.path.exists(depth_file):
        data['colorized_depth'] = np.array(Image.open(depth_file))
    
    # Load batch metadata
    metadata_file = os.path.join(batch_dir, 'batch_metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            data['batch_metadata'] = json.load(f)
    
    return data


def load_batch_data(data_dir, batch_idx):
    """
    Load all data for a specific batch.
    
    Args:
        data_dir: Root directory containing the VGGT output
        batch_idx: Batch index (integer)
    
    Returns:
        dict: Dictionary containing all batch data
    """
    import json
    
    batch_dir = os.path.join(data_dir, f"batch_{batch_idx:03d}")
    if not os.path.exists(batch_dir):
        print(f"Warning: Batch directory {batch_dir} does not exist")
        return {}
    
    data = {'batch_directory': batch_dir, 'batch_index': batch_idx}
    
    # Load batch metadata
    metadata_file = os.path.join(batch_dir, 'batch_metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            data['batch_metadata'] = json.load(f)
            image_names = [os.path.splitext(name)[0] for name in data['batch_metadata'].get('image_names', [])]
    else:
        # Fallback: find image names from ply files
        ply_dir = os.path.join(batch_dir, "ply")
        if os.path.exists(ply_dir):
            ply_files = glob.glob(os.path.join(ply_dir, "*.ply"))
            image_names = [os.path.splitext(os.path.basename(f))[0] for f in ply_files if not f.endswith('combined.ply')]
        else:
            image_names = []
    
    # Load combined point cloud
    combined_ply = os.path.join(batch_dir, "ply", "combined.ply")
    if os.path.exists(combined_ply):
        data['combined_point_cloud'] = trimesh.load(combined_ply)
    
    # Load individual image data
    data['images'] = {}
    for image_name in image_names:
        image_data = load_vggt_data(data_dir, image_name)
        data['images'][image_name] = image_data
    
    return data


def main():
    args = parse_args()
    print("Arguments:", vars(args))
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Load VGGT model with camera and depth heads enabled
    print("Loading VGGT model (for point cloud estimation)...")
    model = VGGT(
        img_size=518,  # Fixed at 518 (model's trained resolution)
        enable_camera=True,
        enable_point=False, # We use depth head + unprojection
        enable_depth=True,
        enable_track=False
    )
    
    # Enable gradient checkpointing for memory savings
    model.train()
    
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state_dict = torch.hub.load_state_dict_from_url(_URL)
    
    # Filter state dict to only load required parameters
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if not any(skip_key in k for skip_key in ['point_head', 'track_head'])
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    print("Model loaded")

    # Get image paths
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    image_path_list = sorted(image_path_list)
    
    # Handle output directory: if not absolute path, make it relative to scene_dir
    if os.path.isabs(args.output_dir):
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.scene_dir, args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Update args.output_dir to use the resolved path for consistency
    args.output_dir = output_dir
    
    print(f"Found {len(image_path_list)} images")

    # Process images in batches to generate point clouds
    process_images_for_pointclouds(model, image_path_list, dtype, args)
    
    print(f"🎉 Point cloud estimation completed successfully!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"🌐 Global point cloud: {os.path.join(args.output_dir, 'pointcloud.ply')}")

    return True


if __name__ == "__main__":
    main() 