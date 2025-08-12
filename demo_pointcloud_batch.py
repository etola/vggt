# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
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
from utils.similarity_transform import compute_similarity_transform, transform_point_cloud_to_colmap_frame


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Batch Point Cloud Estimation")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output point clouds and depth maps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resolution", type=int, default=256, help="Preprocessing resolution. Model always runs at 518.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to process together.")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--conf_threshold", type=float, default=2.0, help="Confidence threshold to filter points (from depth head, >1).")
    parser.add_argument("--colormap", type=str, default="viridis", help="Colormap for depth visualization (e.g., viridis, jet, inferno).")
    
    # Similarity transform arguments
    parser.add_argument("--colmap_sparse_dir", type=str, default=None, help="Path to COLMAP sparse reconstruction for similarity transform alignment")
    parser.add_argument("--align_to_colmap", action="store_true", default=False, help="Enable similarity transform alignment to COLMAP")
    parser.add_argument("--save_vggt_calibration", action="store_true", default=False, help="Save VGGT calibration for transform computation")
    parser.add_argument("--robust_transform", action="store_true", default=False, help="Use robust similarity transform (removes outliers)")
    parser.add_argument("--outlier_threshold", type=float, default=2.0, help="Outlier threshold for robust transform (multiplier of median error)")
    
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
    Run VGGT for a batch of images to get point clouds and depth maps.
    
    Args:
        model: VGGT model with camera and depth heads enabled
        images_batch: [B, 3, H, W] batch of images at model resolution
        dtype: Data type for mixed precision
        vggt_model_resolution: Fixed resolution for VGGT model (518)
    
    Returns:
        points_3d: Numpy array of 3D points [B, H, W, 3]
        depth_conf: Numpy array of depth confidence [B, H, W]
        depth_map: Numpy array of depth maps [B, H, W, 1]
        images_for_color: Torch tensor of images for coloring points [B, 3, H, W]
        extrinsic: Camera extrinsic matrices [B, 3, 4]
        intrinsic: Camera intrinsic matrices [B, 3, 3]
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


def save_vggt_calibration_as_colmap(extrinsics_list, intrinsics_list, image_names_list, output_dir, vggt_model_resolution=518):
    """Save VGGT calibration in COLMAP format for similarity transform computation."""
    import pycolmap
    
    reconstruction = pycolmap.Reconstruction()
    
    # Create cameras (assuming SIMPLE_PINHOLE for simplicity)
    for i, intrinsics_batch in enumerate(intrinsics_list):
        for j, intrinsic in enumerate(intrinsics_batch):
            camera = pycolmap.Camera()
            camera.camera_id = i * len(intrinsics_batch) + j + 1
            camera.model = "SIMPLE_PINHOLE"
            # Extract individual float values from the intrinsic matrix
            f = float(intrinsic[0, 0])    # focal length
            cx = float(intrinsic[0, 2])   # principal point x
            cy = float(intrinsic[1, 2])   # principal point y
            camera.params = [f, cx, cy]
            camera.width = vggt_model_resolution
            camera.height = vggt_model_resolution
            reconstruction.add_camera(camera)
    
    # Create images with poses
    image_id = 1
    for batch_idx, (extrinsics, image_names) in enumerate(zip(extrinsics_list, image_names_list)):
        for i, (ext_matrix, img_name) in enumerate(zip(extrinsics, image_names)):
            # VGGT extrinsics are already in world-to-camera format
            R = ext_matrix[:3, :3]
            t = ext_matrix[:3, 3]
            
            # Create pycolmap Rigid3d object
            cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
            
            # Create image
            camera_id = batch_idx * len(extrinsics) + i + 1
            image = pycolmap.Image(
                id=image_id,
                name=img_name,
                camera_id=camera_id,
                cam_from_world=cam_from_world
            )
            
            # Mark as registered
            image.points2D = pycolmap.ListPoint2D([])
            image.registered = True
            
            reconstruction.add_image(image)
            image_id += 1
    
    # Save in ASCII format
    vggt_calibration_dir = os.path.join(output_dir, "vggt_calibration")
    os.makedirs(vggt_calibration_dir, exist_ok=True)
    reconstruction.write_text(vggt_calibration_dir)
    print(f"VGGT calibration saved to {vggt_calibration_dir}")
    
    return vggt_calibration_dir


def process_images_for_pointclouds(model, image_paths, dtype, args):
    """
    Process images in batches to generate and save point clouds and depth maps.
    """
    vggt_model_resolution = 518
    
    ply_output_dir = os.path.join(args.output_dir, "ply")
    depth_output_dir = os.path.join(args.output_dir, "depth")
    os.makedirs(ply_output_dir, exist_ok=True)
    os.makedirs(depth_output_dir, exist_ok=True)
    
    # For similarity transform
    all_extrinsics = []
    all_intrinsics = []
    all_image_names = []
    
    # Limit number of images if specified
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]
    
    # Split into batches
    num_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
    
    print(f"Processing {len(image_paths)} images in {num_batches} batches of size {args.batch_size}")
    print(f"Preprocessing at {args.resolution}x{args.resolution}, model runs at {vggt_model_resolution}x{vggt_model_resolution}")
    
    # Compute similarity transform if requested
    similarity_transform = None
    if args.align_to_colmap and args.colmap_sparse_dir:
        print("Similarity transform alignment enabled - will compute after first batch")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{num_batches}: images {start_idx + 1}-{end_idx}")
        
        # Load batch with aspect-ratio preservation
        images, _ = load_and_preprocess_images_square(batch_paths, args.resolution)
        
        # Resize to model resolution
        images_for_model = F.interpolate(images, size=(vggt_model_resolution, vggt_model_resolution), mode="bilinear", align_corners=False)
        images_for_model = images_for_model.to(next(model.parameters()).device)

        # Process batch to get point clouds and depth maps
        points_3d_batch, depth_conf_batch, depth_map_batch, images_for_color, extrinsic_batch, intrinsic_batch = run_VGGT_batch_pointcloud(
            model, images_for_model, dtype, vggt_model_resolution
        )
        
        # Store calibration data
        all_extrinsics.append(extrinsic_batch)
        all_intrinsics.append(intrinsic_batch)
        all_image_names.append([os.path.basename(p) for p in batch_paths])
        
        # Compute similarity transform after first batch if requested
        if similarity_transform is None and args.align_to_colmap and args.colmap_sparse_dir and batch_idx == 0:
            # Save current VGGT calibration
            vggt_calibration_dir = save_vggt_calibration_as_colmap(
                [extrinsic_batch], [intrinsic_batch], [[os.path.basename(p) for p in batch_paths]], 
                args.output_dir, vggt_model_resolution
            )
            
            # Compute similarity transform
            print("Computing similarity transform to COLMAP...")
            try:
                if args.robust_transform:
                    print("Using robust similarity transform...")
                    similarity_transform = compute_similarity_transform(
                        vggt_calibration_dir, args.colmap_sparse_dir, verbose=True, use_robust=True
                    )
                else:
                    # Try robust transform first, fall back to regular if needed
                    similarity_transform = compute_similarity_transform(
                        vggt_calibration_dir, args.colmap_sparse_dir, verbose=True, use_robust=True
                    )
                    
                    # Fall back to regular transform if robust doesn't improve much
                    if similarity_transform['rmse'] > 1.0:
                        print("\nRobust transform has high RMSE, trying regular transform...")
                        regular_transform = compute_similarity_transform(
                            vggt_calibration_dir, args.colmap_sparse_dir, verbose=True, use_robust=False
                        )
                        if regular_transform['rmse'] < similarity_transform['rmse']:
                            print("Using regular transform (better RMSE)")
                            similarity_transform = regular_transform
                        else:
                            print("Using robust transform")
                
                # Provide diagnostic information
                rmse = similarity_transform['rmse']
                if rmse > 2.0:
                    print(f"⚠️  WARNING: High RMSE ({rmse:.3f}) indicates poor alignment")
                    print("   This might be due to:")
                    print("   - VGGT and COLMAP using different coordinate systems")
                    print("   - Insufficient camera motion in the batch")
                    print("   - Different scale/units between reconstructions")
                    print("   - VGGT pose estimation issues")
                    if similarity_transform.get('outlier_count', 0) > 0:
                        inliers = similarity_transform.get('inlier_count', 0)
                        outliers = similarity_transform.get('outlier_count', 0)
                        print(f"   - Robust transform used {inliers} inliers, removed {outliers} outliers")
                elif rmse > 0.5:
                    print(f"⚠️  Moderate RMSE ({rmse:.3f}) - alignment may be acceptable")
                else:
                    print(f"✅ Good alignment (RMSE: {rmse:.3f})")
                    
                print(f"Similarity transform computed successfully (RMSE: {similarity_transform['rmse']:.6f})")
            except Exception as e:
                print(f"Warning: Could not compute similarity transform: {e}")
                print("Proceeding without alignment...")
                similarity_transform = None
        
        # --- Save Combined Batch Point Cloud ---
        # Flatten all points, confidences, and colors from the batch
        batch_points_flat = points_3d_batch.reshape(-1, 3)
        batch_conf_flat = depth_conf_batch.flatten()
        batch_colors_np = images_for_color.permute(0, 2, 3, 1).numpy()
        batch_colors_flat = (batch_colors_np.reshape(-1, 3) * 255).astype(np.uint8)

        # Filter the combined points
        combined_conf_mask = batch_conf_flat > args.conf_threshold
        combined_filtered_points = batch_points_flat[combined_conf_mask]
        combined_filtered_colors = batch_colors_flat[combined_conf_mask]
        
        # Apply similarity transform if available
        if similarity_transform is not None:
            combined_filtered_points, combined_filtered_colors = transform_point_cloud_to_colmap_frame(
                combined_filtered_points, combined_filtered_colors, similarity_transform
            )
            suffix = "_aligned"
        else:
            suffix = ""

        # Save the combined point cloud
        combined_ply_path = os.path.join(ply_output_dir, f"batch_{batch_idx:03d}_combined{suffix}.ply")
        if combined_filtered_points.shape[0] > 0:
            combined_pc = trimesh.PointCloud(vertices=combined_filtered_points, colors=combined_filtered_colors)
            combined_pc.export(combined_ply_path)
            print(f"  Saved combined batch with {combined_filtered_points.shape[0]} points to {combined_ply_path}")
        
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
            
            # Get colors for the points
            color_image = images_for_color[i]
            color_image_np = color_image.permute(1, 2, 0).numpy() # HxWx3
            filtered_colors = color_image_np[conf_mask]
            
            # Apply similarity transform if available
            if similarity_transform is not None:
                filtered_points, filtered_colors = transform_point_cloud_to_colmap_frame(
                    filtered_points, (filtered_colors * 255).astype(np.uint8), similarity_transform
                )
                suffix = "_aligned"
            else:
                filtered_colors = (filtered_colors * 255).astype(np.uint8)
                suffix = ""
            
            output_filename = os.path.join(ply_output_dir, f"{file_name_no_ext}{suffix}.ply")
            
            if filtered_points.shape[0] > 0:
                point_cloud = trimesh.PointCloud(vertices=filtered_points, colors=filtered_colors)
                point_cloud.export(output_filename)
                print(f"  Saved {filtered_points.shape[0]} points to {output_filename}")
            else:
                print(f"  No points passed confidence threshold for {output_filename}")
            
            # --- Save Depth Map ---
            frame_depth = depth_map_batch[i]
            colored_depth = colorize_depth_map(frame_depth, cmap=args.colormap)
            
            depth_output_filename = os.path.join(depth_output_dir, f"{file_name_no_ext}_depth.png")
            Image.fromarray(colored_depth).save(depth_output_filename)
            print(f"  Saved depth map to {depth_output_filename}")

        # Aggressive memory cleanup
        del images, images_for_model, points_3d_batch, depth_conf_batch, depth_map_batch, images_for_color
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save complete VGGT calibration if requested
    if args.save_vggt_calibration:
        complete_calibration_dir = save_vggt_calibration_as_colmap(
            all_extrinsics, all_intrinsics, all_image_names, 
            args.output_dir, vggt_model_resolution
        )
        
        # Compute final similarity transform if we have COLMAP data
        if args.colmap_sparse_dir:
            try:
                final_transform = compute_similarity_transform(
                    complete_calibration_dir, args.colmap_sparse_dir, verbose=True
                )
                
                # Save transform results
                import json
                transform_results = {
                    'similarity_transform': {
                        'scale': float(final_transform['scale']),
                        'translation': final_transform['translation'].tolist(),
                        'rotation_matrix': final_transform['rotation'].tolist(),
                    },
                    'rmse': float(final_transform['rmse']),
                    'num_common_images': int(final_transform['num_common_images']),
                    'common_images': final_transform['common_images']
                }
                
                with open(os.path.join(args.output_dir, 'similarity_transform.json'), 'w') as f:
                    json.dump(transform_results, f, indent=2)
                print(f"Similarity transform results saved to {args.output_dir}/similarity_transform.json")
                
            except Exception as e:
                print(f"Warning: Could not compute final similarity transform: {e}")


def main():
    args = parse_args()
    print("Arguments:", vars(args))
    
    # Validate arguments
    if args.align_to_colmap and not args.colmap_sparse_dir:
        raise ValueError("--colmap_sparse_dir must be provided when --align_to_colmap is enabled")
    
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
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Found {len(image_path_list)} images")

    # Process images in batches to generate point clouds
    process_images_for_pointclouds(model, image_path_list, dtype, args)
    
    print(f"Point cloud estimation completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    
    if args.align_to_colmap:
        print("Point clouds have been aligned to COLMAP coordinate system")

    return True


if __name__ == "__main__":
    main() 