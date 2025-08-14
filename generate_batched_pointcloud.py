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
       - By default, each image is used only once across all batches  
       - With --allow_image_reuse, images can appear in multiple batches
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
    │   ├── depth/                   # Colorized depth maps (.png) [only with --save_raw_data]
    │   ├── raw_data/                # Raw depth & confidence (.npy) [only with --save_raw_data]
    │   ├── individual_cameras/      # Camera parameters (.npy) [only with --save_raw_data]
    │   ├── vggt_calibration/        # VGGT calibration (for scale estimation)
    │   ├── transformed/             # Final output in reference coordinate system
    │   │   ├── scale.json           # Scale estimation info
    │   │   ├── cameras.txt          # Reference COLMAP calibration
    │   │   ├── images.txt
    │   │   ├── points3D.txt
    │   │   └── batch.ply            # Combined point cloud [default] OR first image [--save_first_only]
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

    # Save only first image point cloud instead of combined (faster, less storage)
    python3 generate_batched_pointcloud.py -s scene/ -o output/ -g reference_colmap/ --save_first_only

    # Save raw depth data along with point clouds
    python3 generate_batched_pointcloud.py -s scene/ -o output/ -g reference_colmap/ --save_raw_data

    # Allow images to be reused across multiple batches (neighbor-based batching)
    python3 generate_batched_pointcloud.py -s scene/ -o output/ -g reference_colmap/ --allow_image_reuse
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
    estimate_scale_only_from_recons,
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
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save the output point clouds (relative to scene_dir if not absolute)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("-r", "--resolution", type=int, default=518, help="Preprocessing resolution. Model always runs at 518.")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Number of images to process together.")
    parser.add_argument("-m", "--max_images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("-c", "--conf_threshold", type=float, default=2.0, help="Confidence threshold to filter points (from depth head, >1).")
    parser.add_argument("--colormap", type=str, default="viridis", help="Colormap for depth visualization (e.g., viridis, jet, inferno).")
    parser.add_argument("-g", "--reference_calibration", type=str, required=True, help="Directory containing a reference calibration in colmap format")
    parser.add_argument("--use_neighbor_batching", action="store_true", default=True, help="Use neighbor-based batching based on 3D point sharing (default: True)")
    parser.add_argument("--sequential_batching", action="store_true", default=False, help="Force sequential batching instead of neighbor-based (overrides --use_neighbor_batching)")
    parser.add_argument("--allow_image_reuse", action="store_true", default=False, help="Allow images to be reused across multiple batches in neighbor-based batching")

    parser.add_argument("--save_raw_data", action="store_true", default=False, help="Save raw depth and confidence maps as numpy arrays for later use")
    parser.add_argument("--save_first_only", action="store_true", default=False, help="Save only the first image's point cloud instead of the combined point cloud (faster, less storage).")
    
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
    Point clouds are computed separately later using scaled depth maps and reference extrinsics.
    
    Args:
        model: VGGT model with camera and depth heads enabled
        images_batch: [B, 3, H, W] batch of images at model resolution
        dtype: Data type for mixed precision
        vggt_model_resolution: Fixed resolution for VGGT model (518)
    
    Returns:
        depth_conf: Numpy array of depth confidence [B, H, W]
        depth_map: Numpy array of depth maps [B, H, W]
        images_for_color: Torch tensor of images for coloring points [B, 3, H, W]
        extrinsic: VGGT Camera extrinsic matrices [B, 3, 4] (saved for scale estimation)
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

    return depth_conf_np, depth_map_np, images_batch.cpu(), extrinsic_np, intrinsic_np




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


def create_neighbor_based_batches_cached(image_paths, cached_calibration_data, batch_size, allow_reuse=False):
    """
    Create neighbor-based batches using cached calibration data.
    
    Args:
        image_paths: List of image file paths
        cached_calibration_data: Pre-loaded calibration data
        batch_size: Number of images per batch
        allow_reuse: Whether to allow images to be reused across batches
    
    Returns:
        List of batch paths
    """
    point_sharing_info = cached_calibration_data['point_sharing_info']
    return create_neighbor_based_batches(image_paths, point_sharing_info, batch_size, allow_reuse)


def create_neighbor_based_batches(image_paths, point_sharing_info, batch_size, allow_reuse=False):
    """
    Create batches based on 3D point sharing rather than sequential ordering.
    
    Args:
        image_paths: List of all image paths to process
        point_sharing_info: Output from analyze_3d_point_sharing()
        batch_size: Size of each batch
        allow_reuse: Whether to allow images to be reused across batches
    
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
    print(f"  Allow image reuse: {allow_reuse}")
    
    batches = []
    processed_images = set()
    
    # Process images in order, but form batches based on 3D point sharing
    for i, image_name in enumerate(available_images):
        if not allow_reuse and image_name in processed_images:
            continue
        
        print(f"\n🗂️  Creating batch {len(batches) + 1} with target {image_name}:")
        
        # Find best neighbors for this image
        excluded_images = None if allow_reuse else processed_images
        batch_image_names = find_best_neighbors(
            image_name, 
            point_sharing_info, 
            batch_size, 
            excluded_images=excluded_images
        )
        
        # Convert image names back to paths
        batch_paths = []
        for name in batch_image_names:
            if name in name_to_path and (allow_reuse or name not in processed_images):
                batch_paths.append(name_to_path[name])
                if not allow_reuse:
                    processed_images.add(name)
        
        if batch_paths:
            batches.append(batch_paths)
            print(f"    ✅ Batch {len(batches)}: {len(batch_paths)} images")
        
        # Stop if we've processed all images (only when not allowing reuse)
        if not allow_reuse and len(processed_images) >= len(available_images):
            break
    
    print(f"\n📊 Batching Summary:")
    print(f"  Total batches created: {len(batches)}")
    if allow_reuse:
        print(f"  Images processed: {len(available_images)} (reuse allowed)")
        print(f"  Total image instances: {sum(len(batch) for batch in batches)}")
    else:
        print(f"  Images processed: {len(processed_images)}")
        print(f"  Images skipped: {len(image_paths) - len(processed_images)}")
    
    return batches


def load_and_cache_calibration_data(reference_calibration_dir):
    """
    Load and cache all necessary calibration data structures to avoid repeated loading.
    
    Args:
        reference_calibration_dir: Path to reference COLMAP reconstruction
    
    Returns:
        dict: Cached calibration data containing:
            - 'reconstruction': pycolmap.Reconstruction object
            - 'colmap_data': COLMAP calibration data from load_colmap_calibration
            - 'camera_poses': Camera centers and rotations for scale estimation
            - 'point_sharing_info': 3D point sharing analysis for neighbor batching
    """
    print(f"📚 Loading and caching calibration data from {reference_calibration_dir}...")
    
    try:
        # Load COLMAP reconstruction (used by analyze_3d_point_sharing and scale estimation)
        from utils.reconstruction_transform import load_reconstruction, extract_camera_centers_and_rotations
        reconstruction = load_reconstruction(reference_calibration_dir)
        
        # Load COLMAP calibration data (used by load_reference_extrinsics_for_batch)
        from utils.colmap_utils import load_colmap_calibration
        colmap_data = load_colmap_calibration(reference_calibration_dir)
        
        # Extract camera poses (used by scale estimation)
        camera_poses = extract_camera_centers_and_rotations(reconstruction)
        
        # Analyze 3D point sharing (used by neighbor-based batching)
        point_sharing_info = analyze_3d_point_sharing_from_reconstruction(reconstruction)
        
        cached_data = {
            'reconstruction': reconstruction,
            'colmap_data': colmap_data,
            'camera_poses': camera_poses,
            'point_sharing_info': point_sharing_info,
        }
        
        print(f"  ✅ Cached {len(camera_poses)} camera poses")
        print(f"  ✅ Cached point sharing info for {len(point_sharing_info['image_names'])} images")
        
        return cached_data
        
    except Exception as e:
        print(f"  ❌ Error loading calibration data: {e}")
        raise


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


def analyze_3d_point_sharing_from_reconstruction(reconstruction):
    """
    Analyze 3D point sharing between images using a pre-loaded reconstruction.
    
    Args:
        reconstruction: pycolmap.Reconstruction object
    
    Returns:
        dict: Same format as analyze_3d_point_sharing
    """
    from collections import defaultdict
    
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
    
    print(f"  ✅ Analyzed 3D point sharing for {len(image_names)} images")
    print(f"  ✅ Found {len(point_to_images)} 3D points with multi-view tracks")
    
    return {
        'image_to_points': dict(image_to_points),
        'point_to_images': dict(point_to_images), 
        'image_names': image_names
    }


def load_reference_extrinsics_for_batch_cached(batch_image_names, cached_calibration_data):
    """
    Load reference calibration extrinsics for the given batch images using cached data.
    
    Args:
        batch_image_names: List of image names in the batch
        cached_calibration_data: Pre-loaded calibration data from load_and_cache_calibration_data
    
    Returns:
        np.ndarray: Array of extrinsic matrices [B, 3, 4]
    """
    try:
        colmap_data = cached_calibration_data['colmap_data']
        
        if colmap_data is None:
            raise ValueError("Cached COLMAP calibration data is None")
        
        reference_images = colmap_data['images']
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


def compute_pointclouds_with_reference_extrinsics(depth_maps, intrinsics, reference_extrinsics):
    """
    Compute 3D point clouds using VGGT depth maps and intrinsics with reference extrinsics.
    
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
        
        print(f"  ✅ Computed point clouds using reference extrinsics")
        print(f"     Shape: {points_3d.shape}")
        
        return points_3d
        
    except Exception as e:
        print(f"  ❌ Error recomputing point clouds: {e}")
        raise


def process_images_for_pointclouds(model, image_paths, dtype, args, cached_calibration_data):
    """
    Process images in batches to generate and save point clouds.
    Each batch gets its own directory with all related assets.
    
    Batching behavior (neighbor-based):
    - By default: each image used only once across all batches
    - With --allow_image_reuse: images can appear in multiple batches
    
    Point cloud saving behavior:
    - By default: saves combined point cloud from all images in batch as 'batch.ply'
    - With --save_first_only: saves point cloud from first image only as 'batch.ply' (faster, less storage)
    
    Depth maps and raw data are only saved when --save_raw_data is specified.
    """
    vggt_model_resolution = 518
    
    # Limit number of images if specified
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]
    
    # Determine batching strategy
    use_neighbor_batching = args.use_neighbor_batching and not args.sequential_batching
    
    if use_neighbor_batching:
        print(f"🔍 Using cached 3D point sharing analysis...")
        # Use cached 3D point sharing analysis
        point_sharing_info = cached_calibration_data['point_sharing_info']
        
        if point_sharing_info is None:
            print("❌ Failed to get cached point sharing info, falling back to sequential batching")
            use_neighbor_batching = False
        else:
            print(f"✅ Using neighbor-based batching based on cached 3D point sharing")
            batches = create_neighbor_based_batches(image_paths, point_sharing_info, args.batch_size, args.allow_image_reuse)
    
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
        vggt_calibration_dir = os.path.join(batch_dir, "vggt_calibration")
        transformed_dir = os.path.join(batch_dir, "transformed")
        
        if args.save_raw_data:
            depth_output_dir = os.path.join(batch_dir, "depth")
            raw_data_dir = os.path.join(batch_dir, "raw_data")
            individual_cameras_dir = os.path.join(batch_dir, "individual_cameras")
            os.makedirs(depth_output_dir, exist_ok=True)
            os.makedirs(raw_data_dir, exist_ok=True)
            os.makedirs(individual_cameras_dir, exist_ok=True)
        os.makedirs(vggt_calibration_dir, exist_ok=True)
        
        # Optimize image loading when only saving first image
        if args.save_first_only:
            print(f"  🚀 Optimizing for first-image-only processing...")
            # For scale estimation, we need at least 2 images, so process first 2 images minimum
            min_images_for_scale = min(2, len(batch_paths))
            scale_estimation_paths = batch_paths[:min_images_for_scale]
            images, _ = load_and_preprocess_images_square(scale_estimation_paths, args.resolution)
            print(f"  📐 Processing {min_images_for_scale} images for scale estimation, saving point cloud for first only")
        else:
            # Load all batch images (normal behavior)
            images, _ = load_and_preprocess_images_square(batch_paths, args.resolution)
        
        # Resize to model resolution
        images_for_model = F.interpolate(images, size=(vggt_model_resolution, vggt_model_resolution), mode="bilinear", align_corners=False)
        images_for_model = images_for_model.to(next(model.parameters()).device)

        # Get image names for processing (scale estimation vs full batch)
        if args.save_first_only:
            # Process minimum images needed for scale estimation
            min_images_for_scale = min(2, len(batch_paths))
            scale_estimation_paths = batch_paths[:min_images_for_scale]
            scale_estimation_names = [os.path.basename(p) for p in scale_estimation_paths]
            batch_image_names = [os.path.basename(p) for p in batch_paths]  # Keep original for metadata
            
            print(f"  🔄 Loading reference extrinsics for scale estimation images...")
            reference_extrinsics_for_scale = load_reference_extrinsics_for_batch_cached(scale_estimation_names, cached_calibration_data)
            # Only need reference extrinsic for first image for point cloud computation
            reference_extrinsics = reference_extrinsics_for_scale[0:1]
            
        else:
            # Process all batch images (normal behavior)
            batch_image_names = [os.path.basename(p) for p in batch_paths]
            scale_estimation_names = batch_image_names
            
            print(f"  🔄 Loading reference extrinsics for batch images...")
            reference_extrinsics = load_reference_extrinsics_for_batch_cached(batch_image_names, cached_calibration_data)
            reference_extrinsics_for_scale = reference_extrinsics
        
        # Process batch to get depth maps and VGGT intrinsics
        depth_conf_batch, depth_map_batch, images_for_color, vggt_extrinsic_batch, intrinsic_batch = run_VGGT_batch_pointcloud(
            model, images_for_model, dtype, vggt_model_resolution
        )
        
        # Save raw data if requested
        if args.save_raw_data:
            # When save_first_only is True, only save data for the first image
            if args.save_first_only:
                base_name = os.path.splitext(batch_image_names[0])[0]
                
                # Save raw depth and confidence maps for first image only
                depth_file = os.path.join(raw_data_dir, f"{base_name}_depth.npy")
                conf_file = os.path.join(raw_data_dir, f"{base_name}_confidence.npy")
                
                np.save(depth_file, depth_map_batch[0])
                np.save(conf_file, depth_conf_batch[0])
                print(f"  Saved raw depth map to {depth_file}")
                print(f"  Saved confidence map to {conf_file}")
            else:
                # Save for all images
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
        if args.save_first_only:
            # Save scale estimation images' calibration data (minimum 2 for scale estimation)
            min_images_for_scale = min(2, len(batch_paths))
            save_vggt_calibration_as_colmap(
                [vggt_extrinsic_batch[:min_images_for_scale]], [intrinsic_batch[:min_images_for_scale]], [scale_estimation_names], 
                vggt_calibration_dir, vggt_model_resolution
            )
            # Save individual camera parameters (for generate_cloud.py) - using VGGT extrinsics
            if args.save_raw_data:
                save_individual_camera_parameters(vggt_extrinsic_batch[:1], intrinsic_batch[:1], [batch_image_names[0]], individual_cameras_dir)
        else:
            # Save all images' calibration data
            save_vggt_calibration_as_colmap(
                [vggt_extrinsic_batch], [intrinsic_batch], [batch_image_names], 
                vggt_calibration_dir, vggt_model_resolution
            )
            # Save individual camera parameters (for generate_cloud.py) - using VGGT extrinsics
            if args.save_raw_data:
                save_individual_camera_parameters(vggt_extrinsic_batch, intrinsic_batch, batch_image_names, individual_cameras_dir)
        

        # Estimate scale from the batch to the reference calibration using cached data
        # First load the source poses from the saved VGGT calibration
        from utils.reconstruction_transform import load_reconstruction, extract_camera_centers_and_rotations, estimate_scale_only_from_cached_data
        source_reconstruction = load_reconstruction(vggt_calibration_dir)
        source_poses = extract_camera_centers_and_rotations(source_reconstruction)
        
        # Use cached target poses
        target_poses = cached_calibration_data['camera_poses']
        
        scale_result = estimate_scale_only_from_cached_data(source_poses, target_poses, robust_scale=True)
        estimated_scale = scale_result['scale']
        
        print("=== Scale Estimation (source -> target) ===")
        print(f"Common images: {scale_result['num_common']}")
        if scale_result['num_common'] <= 10:
            print(f"Names: {scale_result['common_images']}")
        print(f"Scale: {estimated_scale:.9f}")
        print(f"RMSE estimate (scale-only): {scale_result['rmse_estimate']:.9f}")

        # Save scale information
        os.makedirs(transformed_dir, exist_ok=True)
        scale_json_out = os.path.join(transformed_dir, "scale.json")
        with open(scale_json_out, "w") as f:
            json.dump({
                "scale": float(estimated_scale),
                "rmse_estimate": float(scale_result['rmse_estimate']),
                "num_common": int(scale_result['num_common']),
                "common_images": scale_result['common_images'],
            }, f, indent=2)
        print(f"Wrote scale info to {scale_json_out}")
        
        print(f"  📐 Estimated scale: {estimated_scale:.6f}")
        print(f"  🔄 Scaling depth maps before point cloud computation...")
        
        # Scale the depth maps with the estimated scale
        scaled_depth_maps = depth_map_batch * estimated_scale
        
        # Compute point clouds using scaled depth maps with reference extrinsics  
        if args.save_first_only:
            print(f"  🔄 Computing point cloud for first image only...")
            # Only compute for the first image when save_first_only is enabled
            # Use only first image's data for point cloud computation
            first_depth_only = scaled_depth_maps[0:1]  # [1, H, W]
            first_intrinsic_only = intrinsic_batch[0:1]  # [1, 3, 3]
            points_3d_batch = compute_pointclouds_with_reference_extrinsics(
                first_depth_only, first_intrinsic_only, reference_extrinsics
            )
        else:
            print(f"  🔄 Computing point clouds with scaled depth maps...")
            points_3d_batch = compute_pointclouds_with_reference_extrinsics(
                scaled_depth_maps, intrinsic_batch, reference_extrinsics
            )
        
        print(f"  ✅ Point clouds computed with correct scale and reference poses")
        
        # Create transformed directory 
        os.makedirs(transformed_dir, exist_ok=True)

        # Save reference extrinsics to transformed directory as the final output
        print(f"  💾 Saving reference extrinsics to transformed directory...")
        if args.save_first_only:
            # Only save first image's reference calibration to transformed directory  
            first_intrinsic_only = intrinsic_batch[0:1]  # [1, 3, 3]
            save_vggt_calibration_as_colmap(
                [reference_extrinsics], [first_intrinsic_only], [batch_image_names[0:1]], 
                transformed_dir, vggt_model_resolution
            )
        else:
            save_vggt_calibration_as_colmap(
                [reference_extrinsics], [intrinsic_batch], [batch_image_names], 
                transformed_dir, vggt_model_resolution
            )
        print(f"  ✅ Saved reference calibration as final transformed output")

        # --- Save Combined Batch Point Cloud (only when not save_first_only) ---
        if not args.save_first_only:
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
            
            # Save the combined point cloud directly to transformed directory
            if combined_filtered_points.shape[0] > 0:
                try:
                    transformed_combined_path = os.path.join(transformed_dir, "batch.ply")
                    combined_pc = trimesh.PointCloud(vertices=combined_filtered_points, colors=combined_filtered_colors)
                    combined_pc.export(transformed_combined_path)
                    print(f"  Saved combined batch with {combined_filtered_points.shape[0]} points to {transformed_combined_path}")
                    
                except Exception as e:
                    print(f"  ⚠️  Warning: Failed to save combined point cloud: {e}")
            else:
                print(f"  ⚠️  No points in combined batch, skipping save")
        
        # --- Save Individual Frame Outputs ---
        # Save point cloud: if save_first_only=True, save only first image's point cloud
        if args.save_first_only:
            # When save_first_only is True, we computed point clouds only for the first image
            # The point cloud data is at index 0 in points_3d_batch (which has shape [1, H, W, 3])
            frame_indices = [0]
            base_name = os.path.basename(batch_paths[0])  # Always use the first batch path
        else:
            frame_indices = []  # Don't save individual frames when processing full batch
        
        for i in frame_indices:
            file_name_no_ext = os.path.splitext(base_name)[0]

            # --- Save Point Cloud ---
            frame_points = points_3d_batch[i]       # HxWx3 (always index 0 when save_first_only=True)
            frame_conf = depth_conf_batch[i]         # HxW (use first image's confidence)
            
            # Filter points based on confidence
            conf_mask = frame_conf > args.conf_threshold
            filtered_points = frame_points[conf_mask]
            
            # Load original image for accurate color sampling  
            images_dir = os.path.join(args.scene_dir, "images")
            original_image_path = batch_paths[0] if args.save_first_only else batch_paths[i]
            
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
            
            # Save individual point cloud directly to transformed directory
            if filtered_points.shape[0] > 0:
                try:
                    transformed_output_filename = os.path.join(transformed_dir, "batch.ply")
                    point_cloud = trimesh.PointCloud(vertices=filtered_points, colors=filtered_colors)
                    point_cloud.export(transformed_output_filename)
                    print(f"  Saved {filtered_points.shape[0]} points to {transformed_output_filename}")
                    
                except Exception as e:
                    print(f"  ⚠️  Warning: Failed to save point cloud for {file_name_no_ext}: {e}")
            else:
                print(f"  No points passed confidence threshold for {file_name_no_ext}")
            
            # --- Save Depth Map (if raw data saving is enabled) ---
            if args.save_raw_data:
                frame_depth = depth_map_batch[i]  # Index 0 when save_first_only=True
                colored_depth = colorize_depth_map(frame_depth, cmap=args.colormap)
                
                depth_output_filename = os.path.join(depth_output_dir, f"{file_name_no_ext}_depth.png")
                Image.fromarray(colored_depth).save(depth_output_filename)
                print(f"  Saved depth map to {depth_output_filename}")

        # Save batch-specific metadata
        if args.save_first_only:
            save_batch_metadata(args, [batch_paths[0]], [batch_image_names[0]], batch_dir, batch_idx, use_neighbor_batching)
        else:
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
    Combine all transformed/batch.ply files from all batches into a single global point cloud.
    
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
            transformed_combined_path = os.path.join(batch_dir, "transformed", "batch.ply")
            
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
            'save_first_only': args.save_first_only,
            'vggt_model_resolution': 518,
            'reference_calibration': args.reference_calibration,
            'use_neighbor_batching': args.use_neighbor_batching,
            'sequential_batching': args.sequential_batching,
            'allow_image_reuse': args.allow_image_reuse,
            'actual_batching_used': 'neighbor-based' if use_neighbor_batching else 'sequential'
        },
        'file_structure': {
            'depth_dir': 'depth/' if args.save_raw_data else None,
            'raw_data_dir': 'raw_data/' if args.save_raw_data else None,
            'individual_cameras_dir': 'individual_cameras/' if args.save_raw_data else None,
            'vggt_calibration_dir': 'vggt_calibration/',
            'transformed_dir': 'transformed/'
        },
        'file_formats': {
            'global_point_cloud': 'pointcloud.ply (combined from all batches, in reference frame)',
            'point_clouds': '.ply (trimesh format, in reference frame with correct scale)',
            'batch_point_cloud': 'transformed/batch.ply (batch point cloud)',
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
            'save_first_only': args.save_first_only,
            'vggt_model_resolution': 518,
            'reference_calibration': args.reference_calibration,
            'use_neighbor_batching': args.use_neighbor_batching,
            'sequential_batching': args.sequential_batching,
            'allow_image_reuse': args.allow_image_reuse
        },
        'batch_organization': {
            'total_batches': num_batches,
            'batch_directory_pattern': 'batch_XXX/',
            'description': 'Each batch has its own directory containing all related assets'
        },
        'file_structure_per_batch': {
            'depth_dir': 'depth/' if args.save_raw_data else None,
            'raw_data_dir': 'raw_data/' if args.save_raw_data else None,
            'individual_cameras_dir': 'individual_cameras/' if args.save_raw_data else None,
            'vggt_calibration_dir': 'vggt_calibration/',
            'transformed_dir': 'transformed/',
            'batch_metadata': 'batch_metadata.json'
        },
        'file_formats': {
            'global_point_cloud': 'pointcloud.ply (combined from all batches, in reference frame)',
            'point_clouds': '.ply (trimesh format, in reference frame with correct scale)',
            'batch_point_cloud': 'transformed/batch.ply (batch point cloud)',
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
    
    # Load batch point cloud from transformed directory
    batch_ply_file = os.path.join(batch_dir, "transformed", "batch.ply")
    if os.path.exists(batch_ply_file):
        data['batch_point_cloud'] = trimesh.load(batch_ply_file)
    
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
        # Fallback: try to get image names from metadata or batch file
        image_names = []
    
    # Load batch point cloud from transformed directory
    batch_ply = os.path.join(batch_dir, "transformed", "batch.ply")
    if os.path.exists(batch_ply):
        data['batch_point_cloud'] = trimesh.load(batch_ply)
    
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

    # Load calibration data once for all batches
    cached_calibration_data = load_and_cache_calibration_data(args.reference_calibration)

    # Process images in batches to generate point clouds
    process_images_for_pointclouds(model, image_path_list, dtype, args, cached_calibration_data)
    
    print(f"🎉 Point cloud estimation completed successfully!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"🌐 Global point cloud: {os.path.join(args.output_dir, 'pointcloud.ply')}")

    return True


if __name__ == "__main__":
    main() 