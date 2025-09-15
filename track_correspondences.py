#!/usr/bin/env python3
"""
VGGT Correspondence Tracking Script
==================================

This script uses the VGGT tracking head to track correspondences across frames.
It loads a COLMAP reconstruction, finds nearest frame pairs, and tracks points
using VGGT's tracking capabilities.

Usage:
    python track_correspondences.py -s /path/to/scene -o output_folder

The script expects:
- Images in scene_folder/images/
- COLMAP reconstruction in scene_folder/sparse/
- Output will be saved to scene_folder/output_folder/
"""

import argparse
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import trimesh
from tqdm import tqdm
from PIL import Image

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from colmap_utils import ColmapReconstruction
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VGGT Correspondence Tracking")
    parser.add_argument("-s", "--scene_folder", type=str, required=True, help="Path to scene folder containing images/ and sparse/ directories")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Output folder name (relative to scene_folder)")
    parser.add_argument("--vggt_resolution", type=int, default=518, help="VGGT processing resolution (default: 518)")
    parser.add_argument("--max_tracks_per_frame", type=int, default=1000, help="Maximum number of tracks to initialize per frame")
    parser.add_argument("--min_track_length", type=int, default=3, help="Minimum track length for 3D point initialization")
    parser.add_argument("--min_confidence", type=float, default=0.5, help="Minimum confidence threshold for tracks")
    parser.add_argument("--min_visibility", type=float, default=0.3, help="Minimum visibility threshold for tracks")
    parser.add_argument("--pairs_per_image", type=int, default=8, help="Number of nearest frames to pair with each reference frame")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--ref_image_id", type=int, default=None, help="Process only a specific reference image ID (if not provided, processes all images)")
    parser.add_argument("--max_tracks_vis", type=int, default=50, help="Maximum number of tracks to show in visualization")
    
    return parser.parse_args()


def setup_device_and_dtype(device_arg: str):
    """Setup device and dtype for computation."""
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    return device, dtype


def load_vggt_model(device: str, enable_track: bool = True):
    """Load VGGT model with tracking head enabled."""
    print("Loading VGGT model...")
    
    model = VGGT(
        img_size=518,  # Fixed at 518 (model's trained resolution)
        enable_camera=True,
        enable_point=False,  # We'll use depth head + unprojection
        enable_depth=True,
        enable_track=enable_track
    )
    
    # Load pretrained weights
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state_dict = torch.hub.load_state_dict_from_url(_URL)
    
    # Filter state dict to only load required parameters
    if not enable_track:
        filtered_state_dict = {
            k: v for k, v in state_dict.items() 
            if not any(skip_key in k for skip_key in ['track_head'])
        }
    else:
        filtered_state_dict = state_dict
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print("VGGT model loaded successfully")
    return model


def load_colmap_reconstruction(scene_folder: str):
    """Load COLMAP reconstruction from sparse directory."""
    sparse_dir = os.path.join(scene_folder, "sparse")
    if not os.path.exists(sparse_dir):
        raise ValueError(f"COLMAP sparse directory not found: {sparse_dir}")
    
    print(f"Loading COLMAP reconstruction from {sparse_dir}")
    reconstruction = ColmapReconstruction(sparse_dir)
    
    summary = reconstruction.get_summary()
    print(f"Reconstruction summary:")
    print(f"  - Images: {summary['num_images']}")
    print(f"  - 3D Points: {summary['num_points_3d']}")
    print(f"  - Cameras: {summary['num_cameras']}")
    print(f"  - Average track length: {summary['avg_track_length']:.2f}")
    
    return reconstruction




def resize_images_for_vggt(images: torch.Tensor, target_size: int = 518) -> torch.Tensor:
    """Resize images to VGGT processing resolution."""
    return F.interpolate(images, size=(target_size, target_size), mode="bilinear", align_corners=False)


def compute_scaled_camera_parameters(reconstruction: ColmapReconstruction, 
                                   image_id: int, 
                                   original_size: Tuple[int, int],
                                   vggt_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute scaled camera parameters for VGGT processing."""
    # Get original camera parameters
    K_orig = reconstruction.get_camera_calibration_matrix(image_id)
    cam_from_world = reconstruction.get_image_cam_from_world(image_id)
    
    # Compute scaling factor
    scale_x = vggt_size / original_size[1]  # width
    scale_y = vggt_size / original_size[0]  # height
    
    # Scale the intrinsic matrix
    K_scaled = K_orig.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy
    
    # Extrinsic matrix remains the same
    extrinsic = cam_from_world.matrix()
    
    return extrinsic, K_scaled


def triangulate_points(tracks: np.ndarray, 
                      visibilities: np.ndarray,
                      extrinsics: List[np.ndarray],
                      intrinsics: List[np.ndarray],
                      min_views: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate 3D points from 2D tracks using multiple views."""
    num_tracks, num_views = tracks.shape[:2]
    points_3d = []
    colors = []
    
    for track_idx in range(num_tracks):
        # Get visible views for this track
        visible_views = np.where(visibilities[track_idx] > 0.5)[0]
        
        if len(visible_views) < min_views:
            continue
        
        # Collect 2D points and camera matrices for visible views
        points_2d = []
        P_matrices = []
        
        for i, view_idx in enumerate(visible_views):
            point_2d = tracks[track_idx, view_idx]
            points_2d.append(point_2d)
            
            # Compute projection matrix P = K @ [R|t]
            P = intrinsics[view_idx] @ extrinsics[view_idx]
            P_matrices.append(P)
        
        if len(points_2d) < 2:
            continue
        
        # Triangulate using DLT (Direct Linear Transform)
        try:
            points_2d = np.array(points_2d)
            P_matrices = np.array(P_matrices)
            
            # Build the system of equations A * X = 0
            A = []
            for i in range(len(points_2d)):
                x, y = points_2d[i]
                P = P_matrices[i]
                
                A.append(x * P[2] - P[0])
                A.append(y * P[2] - P[1])
            
            A = np.array(A)
            
            # Solve using SVD
            _, _, V = np.linalg.svd(A)
            X = V[-1]
            
            # Convert from homogeneous coordinates
            if abs(X[3]) > 1e-8:
                point_3d = X[:3] / X[3]
                
                # Check if point is in front of cameras
                is_valid = True
                for i, view_idx in enumerate(visible_views):
                    P = P_matrices[i]
                    z = P[2] @ np.append(point_3d, 1)
                    if z <= 0:
                        is_valid = False
                        break
                
                if is_valid:
                    points_3d.append(point_3d)
                    # Use color from first visible view (placeholder)
                    colors.append([128, 128, 128])  # Gray color
                    
        except np.linalg.LinAlgError:
            continue
    
    if len(points_3d) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    return np.array(points_3d), np.array(colors)


def save_pointcloud(points_3d: np.ndarray, colors: np.ndarray, filename: str):
    """Save 3D points as PLY file."""
    if len(points_3d) == 0:
        print(f"No points to save for {filename}")
        return
    
    # Create trimesh point cloud
    point_cloud = trimesh.PointCloud(vertices=points_3d, colors=colors)
    
    # Save as PLY
    point_cloud.export(filename)
    print(f"Saved {len(points_3d)} points to {filename}")


def get_paired_image_ids(reconstruction: ColmapReconstruction, ref_image_id: int, args) -> List[int]:
    """Get paired image IDs for the reference image."""
    paired_ids = reconstruction._find_best_partner_for_image(
        image_id=ref_image_id,
        min_points=50,  # Lower threshold for more pairs
        parallax_sample_size=100
    )
    
    # Filter out -1 (no match found) and take up to N=8 partners
    valid_partners = [pid for pid in paired_ids if pid != -1]
    return valid_partners[:args.pairs_per_image]  # Take up to pairs_per_image (default 8)


def prepare_image_batch(reconstruction: ColmapReconstruction, 
                       ref_image_id: int, 
                       paired_ids: List[int], 
                       args) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Prepare image batch for VGGT processing."""
    ref_image_path = reconstruction.get_image_path(ref_image_id)
    
    # Get paired image paths
    paired_image_paths = []
    paired_image_ids = []
    
    for paired_id in paired_ids:
        paired_path = reconstruction.get_image_path(paired_id)
        if os.path.exists(paired_path):
            paired_image_paths.append(paired_path)
            paired_image_ids.append(paired_id)
    
    if len(paired_image_paths) == 0:
        raise ValueError("No valid paired images found")
    
    # Prepare image paths for batch processing
    all_image_paths = [ref_image_path] + paired_image_paths
    all_image_ids = [ref_image_id] + paired_image_ids
    
    # Load and preprocess images using the proper batch processing function
    images_tensor, original_coords = load_and_preprocess_images_square(
        all_image_paths, target_size=args.vggt_resolution
    )
    
    return images_tensor, original_coords, all_image_ids


def extract_features_from_resized_image(images_tensor: torch.Tensor, args) -> np.ndarray:
    """Extract good features to track from the resized reference image."""
    # Get the first image (reference image) from the batch
    ref_image_resized = images_tensor[0]  # (3, H, W)
    
    # Convert to numpy and transpose to (H, W, 3) for OpenCV
    ref_image_np = ref_image_resized.permute(1, 2, 0).cpu().numpy()
    
    # Convert from [0, 1] range to [0, 255] range
    ref_image_np = (ref_image_np * 255).astype(np.uint8)
    
    # Extract good features to track from resized reference image
    query_points = get_good_features_to_track(ref_image_np, args.max_tracks_per_frame)
    if len(query_points) == 0:
        raise ValueError("No features found in resized reference image")
    
    return query_points


def get_good_features_to_track(image: np.ndarray, max_points: int = 1000) -> np.ndarray:
    """Extract good features to track using OpenCV."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Parameters for goodFeaturesToTrack
    feature_params = dict(
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=3,
        useHarrisDetector=False,
        k=0.04
    )
    
    # Extract corners
    corners = cv2.goodFeaturesToTrack(gray, **feature_params)
    
    if corners is None:
        return np.array([]).reshape(0, 2)
    
    # Reshape to (N, 2)
    corners = corners.reshape(-1, 2)
    
    return corners


def run_vggt_tracking(model, images_data, query_points, device, dtype) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run VGGT tracking on the image batch."""
    images_tensor, original_coords = images_data
    
    # Query points are already in VGGT resolution, no scaling needed
    # Convert to torch tensor
    query_points_tensor = torch.from_numpy(query_points).float().to(device)
    
    # Run VGGT tracking
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            # Add batch dimension
            images_batch = images_tensor.unsqueeze(0).to(device)  # (1, S, 3, H, W)
            query_points_batch = query_points_tensor.unsqueeze(0)  # (1, N, 2)
            
            # Get aggregated tokens
            aggregated_tokens_list, patch_start_idx = model.aggregator(images_batch)
            
            # Run tracking head
            track_list, vis_scores, conf_scores = model.track_head(
                aggregated_tokens_list, images_batch, patch_start_idx, 
                query_points=query_points_batch
            )
            
            # Get final tracks (from last iteration)
            tracks = track_list[-1]  # (1, S, N, 2)
            visibilities = vis_scores  # (1, S, N)
            confidences = conf_scores  # (1, S, N)
    
    # Convert back to numpy
    tracks = tracks.squeeze(0).cpu().float().numpy()  # (S, N, 2)
    visibilities = visibilities.squeeze(0).cpu().float().numpy()  # (S, N)
    confidences = confidences.squeeze(0).cpu().float().numpy()  # (S, N)
    
    # Keep tracks at VGGT resolution for visualization
    # We'll scale them to original resolution only when needed for triangulation
    return tracks, visibilities, confidences


def filter_good_tracks(tracks: np.ndarray, 
                      visibilities: np.ndarray, 
                      confidences: np.ndarray, 
                      args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter tracks by confidence and visibility criteria."""
    good_tracks = []
    good_visibilities = []
    good_confidences = []
    
    for track_idx in range(tracks.shape[1]):
        track_vis = visibilities[:, track_idx]
        track_conf = confidences[:, track_idx]
        
        # Check if track meets criteria
        avg_confidence = np.mean(track_conf[track_vis > 0.5])
        avg_visibility = np.mean(track_vis)
        visible_frames = np.sum(track_vis > 0.5)
        
        # Require tracks to be visible in at least 3 images for better triangulation
        min_visible_frames = max(3, min(args.min_track_length, tracks.shape[0]))
        
        if (avg_confidence >= args.min_confidence and 
            avg_visibility >= args.min_visibility and 
            visible_frames >= min_visible_frames):
            
            good_tracks.append(tracks[:, track_idx])
            good_visibilities.append(track_vis)
            good_confidences.append(track_conf)
    
    if len(good_tracks) == 0:
        return np.array([]).reshape(0, tracks.shape[0], 2), np.array([]).reshape(0, tracks.shape[0]), np.array([]).reshape(0, tracks.shape[0])
    
    return np.array(good_tracks), np.array(good_visibilities), np.array(good_confidences)


def triangulate_and_save_points(reconstruction: ColmapReconstruction,
                               good_tracks: np.ndarray,
                               good_visibilities: np.ndarray,
                               all_image_ids: List[int],
                               ref_image_name: str,
                               original_coords: torch.Tensor,
                               args) -> bool:
    """Triangulate 3D points and save them as PLY file."""
    if len(good_tracks) == 0:
        print(f"No good tracks found for {ref_image_name}")
        return False
    
    print(f"Found {len(good_tracks)} good tracks for {ref_image_name}")
    
    # Scale tracks to original image resolution for triangulation
    tracks_original_res = good_tracks.copy()
    for frame_idx in range(tracks_original_res.shape[1]):  # Iterate over frames, not tracks
        frame_coords = original_coords[frame_idx].cpu().numpy()  # [6] - coordinate info for this frame
        x1, y1, x2, y2, orig_width, orig_height = frame_coords
        
        # Calculate scale factors from VGGT resolution to original image resolution
        scale_x = orig_width / (x2 - x1)
        scale_y = orig_height / (y2 - y1)
        
        # Scale tracks to original image resolution for this frame
        tracks_original_res[:, frame_idx, 0] = (tracks_original_res[:, frame_idx, 0] - x1) * scale_x
        tracks_original_res[:, frame_idx, 1] = (tracks_original_res[:, frame_idx, 1] - y1) * scale_y
    
    # Filter out tracks that fall in padded regions for visible frames only
    valid_tracks_mask = np.ones(len(tracks_original_res), dtype=bool)
    
    for frame_idx in range(tracks_original_res.shape[1]):
        frame_coords = original_coords[frame_idx].cpu().numpy()  # [6] - coordinate info for this frame
        x1, y1, x2, y2, orig_width, orig_height = frame_coords
        
        # Check if tracks are within original image bounds (not in padding)
        frame_tracks = tracks_original_res[:, frame_idx, :]  # (N_tracks, 2)
        in_bounds = ((frame_tracks[:, 0] >= 0) & (frame_tracks[:, 0] < orig_width) & 
                    (frame_tracks[:, 1] >= 0) & (frame_tracks[:, 1] < orig_height))
        
        # Only check bounds for tracks that are visible in this frame
        frame_visible = good_visibilities[:, frame_idx] > 0.5
        visible_in_bounds = in_bounds | (~frame_visible)  # True if in bounds OR not visible
        
        # Only keep tracks that are in bounds for visible frames
        valid_tracks_mask &= visible_in_bounds
    
    # Filter tracks and visibilities to only include valid ones
    if np.sum(valid_tracks_mask) == 0:
        print(f"No tracks within original image regions for triangulation")
        return False
    
    tracks_original_res = tracks_original_res[valid_tracks_mask]
    good_visibilities = good_visibilities[valid_tracks_mask]
    
    print(f"Filtered to {len(tracks_original_res)} tracks within original image regions for triangulation")
    
    # Compute scaled camera parameters for all views
    extrinsics = []
    intrinsics = []
    
    for img_id in all_image_ids:
        # Get original camera parameters (since tracks are already scaled back to original resolution)
        K_orig = reconstruction.get_camera_calibration_matrix(img_id)
        cam_from_world = reconstruction.get_image_cam_from_world(img_id)
        
        # Use original intrinsics and extrinsics for triangulation
        extrinsics.append(cam_from_world.matrix())
        intrinsics.append(K_orig)
    
    # Triangulate 3D points (require at least 3 views for better accuracy)
    points_3d, colors = triangulate_points(
        tracks_original_res, good_visibilities, extrinsics, intrinsics, min_views=3
    )
    
    # Save point cloud
    output_filename = os.path.join(args.scene_folder, args.output_folder, f"{os.path.splitext(ref_image_name)[0]}_tracks.ply")
    save_pointcloud(points_3d, colors, output_filename)
    
    return True


def visualize_tracks(images_tensor: torch.Tensor,
                    tracks: np.ndarray,
                    visibilities: np.ndarray,
                    all_image_ids: List[int],
                    ref_image_name: str,
                    reconstruction: ColmapReconstruction,
                    original_coords: torch.Tensor,
                    args,
                    max_tracks_to_show: int = 50) -> None:
    """Visualize tracks by creating concatenated image pairs with correspondence lines."""
    output_dir = os.path.join(args.scene_folder, args.output_folder, "track_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensor images to numpy arrays (already resized for VGGT)
    images_np = images_tensor.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)
    images_np = (images_np * 255).astype(np.uint8)  # Convert from [0,1] to [0,255]
    
    # Ensure contiguous memory layout for OpenCV
    images_np = np.ascontiguousarray(images_np)
    
    # Get original image regions (crop out padding)
    ref_coords = original_coords[0].cpu().numpy()  # [x1, y1, x2, y2, orig_width, orig_height]
    ref_x1, ref_y1, ref_x2, ref_y2 = ref_coords[:4].astype(int)
    ref_image_cropped = images_np[0, ref_y1:ref_y2, ref_x1:ref_x2]
    
    # For each paired image, create a visualization
    for i, paired_image_id in enumerate(all_image_ids[1:], 1):
        # Get paired image coordinates and crop
        paired_coords = original_coords[i].cpu().numpy()  # [x1, y1, x2, y2, orig_width, orig_height]
        paired_x1, paired_y1, paired_x2, paired_y2 = paired_coords[:4].astype(int)
        paired_image_cropped = images_np[i, paired_y1:paired_y2, paired_x1:paired_x2]
        
        # Filter tracks that are visible in both images
        # tracks shape: (N_tracks, N_frames, 2)
        # visibilities shape: (N_tracks, N_frames)
        ref_visible = visibilities[:, 0] > 0.5  # Reference image visibility
        paired_visible = visibilities[:, i] > 0.5  # Paired image visibility
        both_visible = ref_visible & paired_visible
        
        if np.sum(both_visible) == 0:
            print(f"No visible tracks between {ref_image_name} and {reconstruction.get_image_name(paired_image_id)}")
            continue
        
        # Get visible tracks
        # tracks[track_idx, frame_idx, :] gives (x, y) for that track in that frame
        ref_points = tracks[both_visible, 0, :].astype(int)  # (N_visible_tracks, 2)
        paired_points = tracks[both_visible, i, :].astype(int)  # (N_visible_tracks, 2) - use frame i for paired image
        
        # Debug: print some track coordinates
        print(f"DEBUG: Sample ref_points (first 3): {ref_points[:3]}")
        print(f"DEBUG: Sample paired_points (first 3): {paired_points[:3]}")
        print(f"DEBUG: Image shapes - ref: {ref_image_cropped.shape}, paired: {paired_image_cropped.shape}")
        print(f"DEBUG: Original coords - ref: {ref_coords}, paired: {paired_coords}")
        
        # Filter tracks that are within the original image regions (not in padding)
        ref_in_bounds = ((ref_points[:, 0] >= ref_x1) & (ref_points[:, 0] < ref_x2) & 
                        (ref_points[:, 1] >= ref_y1) & (ref_points[:, 1] < ref_y2))
        paired_in_bounds = ((paired_points[:, 0] >= paired_x1) & (paired_points[:, 0] < paired_x2) & 
                           (paired_points[:, 1] >= paired_y1) & (paired_points[:, 1] < paired_y2))
        both_in_bounds = ref_in_bounds & paired_in_bounds
        
        if np.sum(both_in_bounds) == 0:
            print(f"No tracks within original image regions between {ref_image_name} and {reconstruction.get_image_name(paired_image_id)}")
            continue
        
        # Filter to only tracks within original image regions
        ref_points = ref_points[both_in_bounds]
        paired_points = paired_points[both_in_bounds]
        
        # Convert from VGGT coordinates to cropped image coordinates
        ref_points_cropped = ref_points.copy().astype(float)
        ref_points_cropped[:, 0] -= ref_x1  # Adjust x coordinates
        ref_points_cropped[:, 1] -= ref_y1  # Adjust y coordinates
        
        paired_points_cropped = paired_points.copy().astype(float)
        paired_points_cropped[:, 0] -= paired_x1  # Adjust x coordinates
        paired_points_cropped[:, 1] -= paired_y1  # Adjust y coordinates
        
        print(f"DEBUG: After cropping - ref_points_cropped (first 3): {ref_points_cropped[:3]}")
        print(f"DEBUG: After cropping - paired_points_cropped (first 3): {paired_points_cropped[:3]}")
        
        # Limit number of tracks to show for better visualization
        if len(ref_points_cropped) > max_tracks_to_show:
            # Randomly sample tracks to show
            np.random.seed(42)  # For reproducible results
            indices = np.random.choice(len(ref_points_cropped), max_tracks_to_show, replace=False)
            ref_points_cropped = ref_points_cropped[indices]
            paired_points_cropped = paired_points_cropped[indices]
        
        
        # Resize images to same height for concatenation
        target_height = max(ref_image_cropped.shape[0], paired_image_cropped.shape[0])
        
        # Resize reference image
        if ref_image_cropped.shape[0] != target_height:
            ref_image_resized = cv2.resize(ref_image_cropped, 
                                         (int(ref_image_cropped.shape[1] * target_height / ref_image_cropped.shape[0]), target_height))
        else:
            ref_image_resized = ref_image_cropped.copy()
        
        # Resize paired image
        if paired_image_cropped.shape[0] != target_height:
            paired_image_resized = cv2.resize(paired_image_cropped, 
                                            (int(paired_image_cropped.shape[1] * target_height / paired_image_cropped.shape[0]), target_height))
        else:
            paired_image_resized = paired_image_cropped.copy()
        
        # Concatenate resized images
        concat_image = np.concatenate([ref_image_resized, paired_image_resized], axis=1)
        
        # Scale points to match resized images
        ref_scale_y = target_height / ref_image_cropped.shape[0]
        paired_scale_y = target_height / paired_image_cropped.shape[0]
        
        ref_points_final = ref_points_cropped.copy().astype(float)
        ref_points_final[:, 1] *= ref_scale_y  # Scale y coordinates
        
        paired_points_final = paired_points_cropped.copy().astype(float)
        paired_points_final[:, 1] *= paired_scale_y  # Scale y coordinates
        paired_points_final[:, 0] += ref_image_resized.shape[1]  # Offset by reference image width
        
        # Draw correspondence lines
        for ref_pt, paired_pt in zip(ref_points_final, paired_points_final):
            # Draw circles at feature points
            cv2.circle(concat_image, tuple(ref_pt.astype(int)), 3, (0, 255, 0), -1)
            cv2.circle(concat_image, tuple(paired_pt.astype(int)), 3, (0, 255, 0), -1)
            
            # Draw line between corresponding points
            cv2.line(concat_image, tuple(ref_pt.astype(int)), tuple(paired_pt.astype(int)), (255, 0, 0), 1)
        
        # Save visualization
        paired_image_name = reconstruction.get_image_name(paired_image_id)
        output_filename = f"{ref_image_name}_{paired_image_name}_tracks.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR))
        print(f"Saved track visualization: {output_filename} ({len(ref_points)} correspondences shown, max {max_tracks_to_show})")


def process_reference_view(reconstruction: ColmapReconstruction, 
                          ref_image_id: int, 
                          model, 
                          device: str, 
                          dtype: torch.dtype, 
                          args) -> bool:
    """
    Process tracking for a specific reference view.
    
    Args:
        reconstruction: COLMAP reconstruction object
        ref_image_id: ID of the reference image to process
        model: VGGT model
        device: Device to use for computation
        dtype: Data type for computation
        args: Command line arguments
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Get paired image IDs
        paired_ids = get_paired_image_ids(reconstruction, ref_image_id, args)
        
        if len(paired_ids) == 0:
            ref_image_name = reconstruction.get_image_name(ref_image_id)
            print(f"No pairs found for {ref_image_name}")
            return False
        
        # Get reference image name
        ref_image_name = reconstruction.get_image_name(ref_image_id)
        
        # Prepare image batch
        images_tensor, original_coords, all_image_ids = prepare_image_batch(
            reconstruction, ref_image_id, paired_ids, args
        )
        
        # Extract features from resized reference image
        query_points = extract_features_from_resized_image(images_tensor, args)
        
        print(f"Processing {ref_image_name} with {len(query_points)} query points and {len(paired_ids)} paired frames")
        
        # Run VGGT tracking
        tracks, visibilities, confidences = run_vggt_tracking(
            model, (images_tensor, original_coords), query_points, device, dtype
        )
        
        # Filter good tracks
        good_tracks, good_visibilities, good_confidences = filter_good_tracks(
            tracks, visibilities, confidences, args
        )
        
        # Triangulate and save points
        success = triangulate_and_save_points(
            reconstruction, good_tracks, good_visibilities, all_image_ids, ref_image_name, original_coords, args
        )
        
        # Visualize tracks
        visualize_tracks(images_tensor, good_tracks, good_visibilities, all_image_ids, ref_image_name, reconstruction, original_coords, args, args.max_tracks_vis)
        
        return success
        
    except Exception as e:
        print(f"Error processing reference view {ref_image_id}: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device and dtype
    device, dtype = setup_device_and_dtype(args.device)
    
    # Load VGGT model
    model = load_vggt_model(device, enable_track=True)
    
    # Load COLMAP reconstruction (images directory will be inferred automatically)
    reconstruction = load_colmap_reconstruction(args.scene_folder)
    
    # Create output directory
    output_dir = os.path.join(args.scene_folder, args.output_folder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process reference frame(s)
    if args.ref_image_id is not None:
        # Process only the specified image ID
        if args.ref_image_id not in reconstruction.get_all_image_ids():
            print(f"Error: Image ID {args.ref_image_id} not found in reconstruction")
            return
        
        print(f"Processing specific image ID: {args.ref_image_id}")
        success = process_reference_view(
            reconstruction, args.ref_image_id, model, device, dtype, args
        )
        if success:
            print("Processing completed successfully!")
        else:
            print("Processing failed!")
    else:
        # Process all reference frames
        for ref_image_id in tqdm(reconstruction.get_all_image_ids(), desc="Processing frames"):
            success = process_reference_view(
                reconstruction, ref_image_id, model, device, dtype, args
            )
            if not success:
                continue
            break
    print("Tracking completed!")


if __name__ == "__main__":
    main()
