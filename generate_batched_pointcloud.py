# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT Batch Point Cloud Generation

Generate point clouds from image sequences using VGGT model with batched processing.

Examples:
    # Basic usage with short flags (output relative to scene directory)
    python3 generate_batched_pointcloud.py -s scene/ -o output/

    # Specify batch size and resolution
    python3 generate_batched_pointcloud.py -s scene/ -o results/ -b 16 -r 512

    # Use absolute output path
    python3 generate_batched_pointcloud.py -s scene/ -o /tmp/pointclouds/

    # Limit number of images and set confidence threshold
    python3 generate_batched_pointcloud.py -s scene/ -o output/ -m 50 -c 1.5

    # Process with custom settings
    python3 generate_batched_pointcloud.py -s scene/ -o output/ -b 4 -c 2.5 --colormap jet
"""

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
from utils.colmap_utils import save_vggt_calibration_as_colmap, save_individual_camera_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Batch Point Cloud Estimation")
    parser.add_argument("-s", "--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save the output point clouds and depth maps (relative to scene_dir if not absolute)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("-r", "--resolution", type=int, default=518, help="Preprocessing resolution. Model always runs at 518.")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Number of images to process together.")
    parser.add_argument("-m", "--max_images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("-c", "--conf_threshold", type=float, default=2.0, help="Confidence threshold to filter points (from depth head, >1).")
    parser.add_argument("--colormap", type=str, default="viridis", help="Colormap for depth visualization (e.g., viridis, jet, inferno).")
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


def process_images_for_pointclouds(model, image_paths, dtype, args):
    """
    Process images in batches to generate and save point clouds and depth maps.
    """
    vggt_model_resolution = 518
    
    ply_output_dir = os.path.join(args.output_dir, "ply")
    depth_output_dir = os.path.join(args.output_dir, "depth")
    raw_data_dir = os.path.join(args.output_dir, "raw_data")
    colmap_dir = os.path.join(args.output_dir, "colmap_calibration")
    
    os.makedirs(ply_output_dir, exist_ok=True)
    os.makedirs(depth_output_dir, exist_ok=True)
    if args.save_raw_data:
        os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(colmap_dir, exist_ok=True)
    
    # Limit number of images if specified
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]
    
    # Split into batches
    num_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
    
    print(f"Processing {len(image_paths)} images in {num_batches} batches of size {args.batch_size}")
    print(f"Preprocessing at {args.resolution}x{args.resolution}, model runs at {vggt_model_resolution}x{vggt_model_resolution}")
    
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
        
        # Get batch image names for similarity transform and saving
        batch_image_names = [os.path.basename(p) for p in batch_paths]
        
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
        
        # Save COLMAP format calibration
        colmap_batch_dir = os.path.join(colmap_dir, f"batch_{batch_idx:03d}")
        save_vggt_calibration_as_colmap(
            [extrinsic_batch], [intrinsic_batch], [batch_image_names], 
            colmap_batch_dir, vggt_model_resolution
        )
        
        # Save individual camera parameters (for generate_cloud.py)
        individual_cameras_dir = os.path.join(args.output_dir, "individual_cameras")
        if args.save_raw_data:
            save_individual_camera_parameters(extrinsic_batch, intrinsic_batch, batch_image_names, individual_cameras_dir)
        
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
        
        # Save the combined point cloud
        combined_ply_path = os.path.join(ply_output_dir, f"batch_{batch_idx:03d}_combined.ply")
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
            filtered_colors = (filtered_colors * 255).astype(np.uint8)
            
            # Save individual point cloud
            output_filename = os.path.join(ply_output_dir, f"{file_name_no_ext}.ply")
            point_cloud = trimesh.PointCloud(vertices=filtered_points, colors=filtered_colors)
            point_cloud.export(output_filename)
            print(f"  Saved {filtered_points.shape[0]} points to {output_filename}")
            
            if filtered_points.shape[0] == 0:
                print(f"  No points passed confidence threshold for {file_name_no_ext}")
            
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
    
    # Save processing metadata
    save_processing_metadata(args, image_paths, args.output_dir)


def save_processing_metadata(args, image_paths, output_dir):
    """Save metadata about the processing parameters and file structure."""
    import json
    
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
            'vggt_model_resolution': 518
        },
        'file_structure': {
            'ply_dir': 'ply/',
            'depth_dir': 'depth/',
            'raw_data_dir': 'raw_data/' if args.save_raw_data else None,
            'cameras_dir': 'cameras/',
            'colmap_calibration_dir': 'colmap_calibration/'
        },
        'file_formats': {
            'point_clouds': '.ply (trimesh format)',
            'depth_maps': '.png (colorized visualization)',
            'raw_depth': '.npy (numpy array, float32)',
            'confidence': '.npy (numpy array, float32)',
            'extrinsics': '.npy (numpy array, shape [3, 4])',
            'intrinsics': '.npy (numpy array, shape [3, 3])',
            'colmap_calibration': 'COLMAP sparse reconstruction format (cameras.txt, images.txt, points3D.txt)'
        },
        'data_info': {
            'total_images_processed': len(image_paths),
            'image_names': [os.path.basename(p) for p in image_paths]
        }
    }
    

    
    metadata_file = os.path.join(output_dir, 'processing_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Processing metadata saved to {metadata_file}")


def load_vggt_data(data_dir, image_name):
    """
    Load VGGT data for a specific image.
    
    Args:
        data_dir: Directory containing the VGGT output
        image_name: Name of the image (without extension)
    
    Returns:
        dict: Dictionary containing loaded data
    """
    from utils.colmap_utils import load_individual_camera_parameters
    
    data = {}
    
    # Load raw depth and confidence maps
    raw_data_dir = os.path.join(data_dir, "raw_data")
    if os.path.exists(raw_data_dir):
        depth_file = os.path.join(raw_data_dir, f"{image_name}_depth.npy")
        conf_file = os.path.join(raw_data_dir, f"{image_name}_confidence.npy")
        
        if os.path.exists(depth_file):
            data['depth_map'] = np.load(depth_file)
        if os.path.exists(conf_file):
            data['confidence_map'] = np.load(conf_file)
    
    # Load camera parameters (try numpy arrays first, then COLMAP format)
    cameras_dir = os.path.join(data_dir, "cameras")
    if os.path.exists(cameras_dir):
        camera_data = load_individual_camera_parameters(image_name, cameras_dir)
        if camera_data:
            data.update(camera_data)
    

    
    # Load point cloud
    ply_dir = os.path.join(data_dir, "ply")
    ply_file = os.path.join(ply_dir, f"{image_name}.ply")
    if os.path.exists(ply_file):
        data['point_cloud'] = trimesh.load(ply_file)
    
    # Load colorized depth map
    depth_dir = os.path.join(data_dir, "depth")
    depth_file = os.path.join(depth_dir, f"{image_name}_depth.png")
    if os.path.exists(depth_file):
        data['colorized_depth'] = np.array(Image.open(depth_file))
    
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
    
    print(f"Point cloud estimation completed successfully!")
    print(f"Results saved to: {args.output_dir}")

    return True


if __name__ == "__main__":
    main() 