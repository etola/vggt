# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dense Feature Correspondence and Triangulation

This script generates dense point clouds by:
1. Selecting image pairs based on 3D point sharing and baseline requirements
2. Extracting dense features using VGGT's DPT head
3. Matching features between pairs using reciprocal nearest neighbors
4. Triangulating correspondences to generate point clouds
5. Filtering erroneous points using epipolar geometry and baseline constraints
6. Merging all pair point clouds into a single dense reconstruction

The script handles the coordinate transformations between full-resolution COLMAP 
calibration and VGGT's preprocessed model resolution (518x518 with padding).

Usage:
    python extract_dense_correspondences.py -s scene/ -g reference_colmap/ -o output/
    
Output:
    - Individual pair point clouds in pair_xxx/ directories  
    - Global merged point cloud as dense_pointcloud.ply
    - Processing metadata and statistics
"""

import os
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import traceback
import trimesh
from PIL import Image
from collections import defaultdict, Counter
import gc

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.geometry import unproject_depth_map_to_point_map

# Utilities
from utils.reconstruction_transform import load_reconstruction
from utils.colmap_utils import load_colmap_calibration

# Dense correlation-based matching (replacing fast_nn)
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Dense Feature Correspondence and Triangulation")
    parser.add_argument("-s", "--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("-g", "--reference_calibration", type=str, required=True, help="Directory containing reference COLMAP calibration")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save output point clouds")
    
    # Processing parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("-r", "--resolution", type=int, default=518, help="Preprocessing resolution for VGGT model")
    parser.add_argument("--max_pairs", type=int, default=None, help="Maximum number of pairs to process")
    parser.add_argument("--min_shared_points", type=int, default=100, help="Minimum shared 3D points between image pairs")
    parser.add_argument("--min_baseline_ratio", type=float, default=0.1, help="Minimum baseline as ratio of scene size")
    
    # Feature matching parameters
    parser.add_argument("--subsample_step", type=int, default=4, help="Subsampling step for dense features (for efficiency)")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for feature matching")
    parser.add_argument("--max_correspondences", type=int, default=10000, help="Maximum correspondences per pair (for efficiency)")
    
    # Filtering parameters
    parser.add_argument("--epipolar_threshold", type=float, default=1.0, help="Epipolar line distance threshold in pixels")
    parser.add_argument("--min_triangulation_angle", type=float, default=2.0, help="Minimum triangulation angle in degrees")
    parser.add_argument("--max_reprojection_error", type=float, default=4.0, help="Maximum reprojection error in pixels")
    
    return parser.parse_args()


def analyze_3d_point_sharing_for_pairs(reconstruction):
    """
    Analyze 3D point sharing between images for pair selection.
    
    Returns:
        dict: {
            'image_to_points': dict mapping image names to sets of 3D point IDs,
            'point_to_images': dict mapping 3D point IDs to sets of image names,
            'image_names': list of all image names,
            'image_positions': dict mapping image names to camera centers
        }
    """
    image_to_points = defaultdict(set)
    point_to_images = defaultdict(set)
    image_names = []
    image_positions = {}
    
    # Get all registered image names and positions
    for image_id, image in reconstruction.images.items():
        if image.registered:
            image_names.append(image.name)
            # Get camera center from extrinsics using correct pycolmap API
            cam_from_world = image.cam_from_world
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation
            camera_center = -R.T @ t
            image_positions[image.name] = camera_center
    
    # Analyze 3D point tracks
    for point3d_id, point3d in reconstruction.points3D.items():
        track = point3d.track
        
        for track_element in track.elements:
            image_id = track_element.image_id
            
            if image_id in reconstruction.images:
                image = reconstruction.images[image_id]
                if image.registered:
                    image_name = image.name
                    image_to_points[image_name].add(point3d_id)
                    point_to_images[point3d_id].add(image_name)
    
    print(f"📊 Point sharing analysis:")
    print(f"  Total registered images: {len(image_names)}")
    print(f"  Total 3D points: {len(reconstruction.points3D)}")
    
    return {
        'image_to_points': dict(image_to_points),
        'point_to_images': dict(point_to_images),
        'image_names': sorted(image_names),
        'image_positions': image_positions
    }


def calculate_baseline_distance(pos1, pos2):
    """Calculate Euclidean distance between two camera positions."""
    return np.linalg.norm(pos1 - pos2)


def find_good_image_pairs(point_sharing_info, min_shared_points=100, min_baseline_ratio=0.1):
    """
    Find image pairs with sufficient 3D point overlap and baseline.
    
    Args:
        point_sharing_info: Output from analyze_3d_point_sharing_for_pairs
        min_shared_points: Minimum number of shared 3D points
        min_baseline_ratio: Minimum baseline as ratio of scene size
    
    Returns:
        list: List of (image1, image2, shared_count, baseline) tuples
    """
    image_to_points = point_sharing_info['image_to_points']
    image_positions = point_sharing_info['image_positions']
    image_names = point_sharing_info['image_names']
    
    # Calculate scene size for baseline normalization
    all_positions = np.array(list(image_positions.values()))
    scene_size = np.linalg.norm(all_positions.max(axis=0) - all_positions.min(axis=0))
    min_baseline = min_baseline_ratio * scene_size
    
    print(f"🎯 Finding good image pairs:")
    print(f"  Scene size: {scene_size:.3f}")
    print(f"  Min baseline: {min_baseline:.3f}")
    print(f"  Min shared points: {min_shared_points}")
    
    good_pairs = []
    
    for i, image1 in enumerate(image_names):
        for j, image2 in enumerate(image_names[i+1:], i+1):
            # Check shared 3D points
            points1 = image_to_points.get(image1, set())
            points2 = image_to_points.get(image2, set())
            shared_points = len(points1.intersection(points2))
            
            if shared_points < min_shared_points:
                continue
            
            # Check baseline
            pos1 = image_positions[image1]
            pos2 = image_positions[image2]
            baseline = calculate_baseline_distance(pos1, pos2)
            
            if baseline < min_baseline:
                continue
            
            good_pairs.append((image1, image2, shared_points, baseline))
    
    # Sort by number of shared points (descending)
    good_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"  Found {len(good_pairs)} good pairs")
    if len(good_pairs) > 0:
        print(f"  Best pair: {good_pairs[0][0]} - {good_pairs[0][1]} ({good_pairs[0][2]} shared points, baseline: {good_pairs[0][3]:.3f})")
    
    return good_pairs


def adjust_intrinsics_for_preprocessing(intrinsics, original_size, target_size=518):
    """
    Adjust camera intrinsics to account for VGGT preprocessing (padding + scaling).
    
    Args:
        intrinsics: [3, 3] camera intrinsic matrix for original image
        original_size: (width, height) of original image
        target_size: Target size for VGGT model (default 518)
    
    Returns:
        np.ndarray: Adjusted intrinsic matrix [3, 3]
    """
    width, height = original_size
    
    # Calculate padding and scaling (same as load_and_preprocess_images_square)
    max_dim = max(width, height)
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    scale = target_size / max_dim
    
    # Adjust intrinsics
    adjusted_intrinsics = intrinsics.copy()
    
    # Adjust for padding
    adjusted_intrinsics[0, 2] += left  # cx
    adjusted_intrinsics[1, 2] += top   # cy
    
    # Adjust for scaling
    adjusted_intrinsics[0, 0] *= scale  # fx
    adjusted_intrinsics[1, 1] *= scale  # fy
    adjusted_intrinsics[0, 2] *= scale  # cx
    adjusted_intrinsics[1, 2] *= scale  # cy
    
    return adjusted_intrinsics


def extract_dense_features(model, images_batch, dtype):
    """
    Extract dense features using VGGT's DPT head in feature_only mode.
    
    Args:
        model: VGGT model with DPT head enabled
        images_batch: [B, 3, H, W] batch of preprocessed images
        dtype: Data type for mixed precision
    
    Returns:
        torch.Tensor: Dense features [B, D, H, W]
    """
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            # Expected input shape for aggregator: [1, B, 3, H, W]
            vggt_input = images_batch.unsqueeze(0)
            
            aggregated_tokens_list, ps_idx = model.aggregator(vggt_input)
            
            # Extract dense features using DPT head in feature_only mode
            # This returns dense feature maps after the scratch_forward fusion
            dense_features = model.depth_head(aggregated_tokens_list, images=vggt_input, patch_start_idx=ps_idx)
            
    return dense_features.squeeze(0)  # Remove outer batch dimension [B, D, H, W]


def triangulate_correspondences(pts1, pts2, K1, K2, R1, t1, R2, t2):
    """
    Triangulate 3D points from correspondences using camera parameters.
    
    Args:
        pts1, pts2: [N, 2] corresponding points in images
        K1, K2: [3, 3] camera intrinsic matrices
        R1, t1: Camera 1 rotation [3, 3] and translation [3]
        R2, t2: Camera 2 rotation [3, 3] and translation [3]
    
    Returns:
        np.ndarray: [N, 3] triangulated 3D points
    """
    # Construct projection matrices
    P1 = K1 @ np.hstack([R1, t1.reshape(-1, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(-1, 1)])
    
    # Triangulate using DLT
    points_3d = []
    
    for i in range(len(pts1)):
        # Create system of equations Ax = 0
        A = np.array([
            pts1[i, 0] * P1[2] - P1[0],
            pts1[i, 1] * P1[2] - P1[1],
            pts2[i, 0] * P2[2] - P2[0],
            pts2[i, 1] * P2[2] - P2[1]
        ])
        
        # Solve using SVD
        try:
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            
            # Check for valid homogeneous coordinate
            if np.abs(X[3]) < 1e-10:
                # Invalid triangulation, set to a default point
                X = np.array([0, 0, 1, 1])
            
            X = X[:3] / X[3]  # Convert from homogeneous
            
            # Check for points behind cameras (negative Z in camera coordinates)
            # Transform to camera 1 coordinates for checking
            pt_cam1 = R1 @ X + t1
            if pt_cam1[2] < 0:
                # Point is behind camera, set to a reasonable default
                X = np.array([0, 0, 1])
                
            points_3d.append(X)
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback for numerical issues
            points_3d.append(np.array([0, 0, 1]))
    
    return np.array(points_3d)


def compute_epipolar_distance(pts1, pts2, F):
    """
    Compute epipolar distance for point correspondences.
    
    Args:
        pts1, pts2: [N, 2] corresponding points
        F: [3, 3] fundamental matrix
    
    Returns:
        np.ndarray: [N] epipolar distances
    """
    # Convert to homogeneous coordinates
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
    
    # Compute epipolar lines
    lines2 = (F @ pts1_h.T).T
    lines1 = (F.T @ pts2_h.T).T
    
    # Compute point-to-line distances
    dist1 = np.abs(np.sum(lines1 * pts1_h, axis=1)) / np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)
    dist2 = np.abs(np.sum(lines2 * pts2_h, axis=1)) / np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2)
    
    return np.maximum(dist1, dist2)


def compute_triangulation_angle(pts_3d, cam_center1, cam_center2):
    """
    Compute triangulation angle for 3D points.
    
    Args:
        pts_3d: [N, 3] triangulated 3D points
        cam_center1, cam_center2: [3] camera centers
    
    Returns:
        np.ndarray: [N] triangulation angles in degrees
    """
    # Vectors from cameras to points
    vec1 = pts_3d - cam_center1[None, :]
    vec2 = pts_3d - cam_center2[None, :]
    
    # Normalize vectors
    vec1_norm = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8)
    
    # Compute angles
    cos_angles = np.sum(vec1_norm * vec2_norm, axis=1)
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles) * 180 / np.pi
    
    return angles


def compute_reprojection_error(pts_3d, pts1, pts2, K1, K2, R1, t1, R2, t2):
    """
    Compute reprojection error for triangulated points.
    
    Args:
        pts_3d: [N, 3] triangulated 3D points
        pts1, pts2: [N, 2] original 2D correspondences
        K1, K2, R1, t1, R2, t2: Camera parameters
    
    Returns:
        np.ndarray: [N] reprojection errors
    """
    # Project 3D points to cameras
    pts_3d_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    
    P1 = K1 @ np.hstack([R1, t1.reshape(-1, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(-1, 1)])
    
    proj1_h = (P1 @ pts_3d_h.T).T
    proj2_h = (P2 @ pts_3d_h.T).T
    
    proj1 = proj1_h[:, :2] / proj1_h[:, 2:3]
    proj2 = proj2_h[:, :2] / proj2_h[:, 2:3]
    
    # Compute errors
    error1 = np.linalg.norm(proj1 - pts1, axis=1)
    error2 = np.linalg.norm(proj2 - pts2, axis=1)
    
    return np.maximum(error1, error2)


def filter_correspondences(pts1, pts2, pts_3d, K1, K2, R1, t1, R2, t2, 
                         cam_center1, cam_center2, args):
    """
    Filter correspondences using multiple geometric constraints.
    
    Returns:
        np.ndarray: Boolean mask of valid correspondences
    """
    num_points = len(pts1)
    valid_mask = np.ones(num_points, dtype=bool)
    
    print(f"    🔍 Filtering {num_points} correspondences...")
    
    # 1. Epipolar constraint
    if args.epipolar_threshold > 0:
        try:
            # Compute fundamental matrix
            t_diff = t2 - t1
            # Create skew-symmetric matrix for cross product
            t_skew = np.array([
                [0, -t_diff[2], t_diff[1]],
                [t_diff[2], 0, -t_diff[0]],
                [-t_diff[1], t_diff[0], 0]
            ])
            E = t_skew @ (R2 @ R1.T)  # Essential matrix
            F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)  # Fundamental matrix
            
            epipolar_distances = compute_epipolar_distance(pts1, pts2, F)
            epipolar_mask = epipolar_distances < args.epipolar_threshold
            valid_mask &= epipolar_mask
            
            print(f"      Epipolar filter: {np.sum(epipolar_mask)}/{num_points} points")
        except Exception as e:
            print(f"      Warning: Epipolar filtering failed: {e}")
    
    # 2. Triangulation angle constraint
    if args.min_triangulation_angle > 0:
        angles = compute_triangulation_angle(pts_3d, cam_center1, cam_center2)
        angle_mask = angles > args.min_triangulation_angle
        valid_mask &= angle_mask
        
        print(f"      Angle filter: {np.sum(angle_mask)}/{np.sum(valid_mask)} points")
    
    # 3. Reprojection error constraint
    if args.max_reprojection_error > 0:
        reproj_errors = compute_reprojection_error(pts_3d, pts1, pts2, K1, K2, R1, t1, R2, t2)
        reproj_mask = reproj_errors < args.max_reprojection_error
        valid_mask &= reproj_mask
        
        print(f"      Reprojection filter: {np.sum(reproj_mask)}/{np.sum(valid_mask)} points")
    
    print(f"    ✅ Final valid points: {np.sum(valid_mask)}/{num_points}")
    
    return valid_mask


def compute_dense_correlations(features1, features2, search_radius=8):
    """
    Compute dense correlations between two feature maps.
    
    Args:
        features1: [D, H, W] feature map from first image
        features2: [D, H, W] feature map from second image  
        search_radius: Radius for local search window
    
    Returns:
        correlations: [H, W, 2*radius+1, 2*radius+1] correlation maps
        coords: [H, W, 2] coordinate grid for features1
    """
    D, H, W = features1.shape
    device = features1.device
    
    # Create coordinate grid for first image
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    coords1 = torch.stack([x_coords, y_coords], dim=-1)  # [H, W, 2]
    
    # Normalize features
    features1_norm = torch.nn.functional.normalize(features1, p=2, dim=0)  # [D, H, W]
    features2_norm = torch.nn.functional.normalize(features2, p=2, dim=0)  # [D, H, W]
    
    # Create search offsets
    r = search_radius
    dy, dx = torch.meshgrid(
        torch.arange(-r, r+1, device=device),
        torch.arange(-r, r+1, device=device),
        indexing='ij'
    )
    search_offsets = torch.stack([dx, dy], dim=-1)  # [2r+1, 2r+1, 2]
    
    # Compute correlations for each pixel
    correlations = torch.zeros(H, W, 2*r+1, 2*r+1, device=device)
    
    for i, offset in enumerate(search_offsets.reshape(-1, 2)):
        offset_x, offset_y = offset[0].item(), offset[1].item()
        
        # Create shifted coordinate grid
        shifted_coords = coords1.clone()
        shifted_coords[:, :, 0] += offset_x  # x offset
        shifted_coords[:, :, 1] += offset_y  # y offset
        
        # Clamp to valid image bounds
        shifted_coords[:, :, 0] = torch.clamp(shifted_coords[:, :, 0], 0, W-1)
        shifted_coords[:, :, 1] = torch.clamp(shifted_coords[:, :, 1], 0, H-1)
        
        # Sample features from second image at shifted locations
        shifted_x = shifted_coords[:, :, 0].long()
        shifted_y = shifted_coords[:, :, 1].long()
        
        sampled_features = features2_norm[:, shifted_y, shifted_x]  # [D, H, W]
        
        # Compute correlation (cosine similarity)
        correlation = torch.sum(features1_norm * sampled_features, dim=0)  # [H, W]
        
        # Store in correlation map
        oy, ox = divmod(i, 2*r+1)
        correlations[:, :, oy, ox] = correlation
    
    return correlations, coords1


def compute_epipolar_correlations(features1, features2, K1, K2, R1, t1, R2, t2, search_radius=4):
    """
    Compute correlations constrained to epipolar lines.
    
    Args:
        features1, features2: [D, H, W] feature maps
        K1, K2: Camera intrinsics [3, 3]
        R1, t1, R2, t2: Camera extrinsics 
        search_radius: Search radius along epipolar line
    
    Returns:
        correspondences: [N, 4] array of [x1, y1, x2, y2] correspondences
        confidences: [N] confidence scores
    """
    D, H, W = features1.shape
    device = features1.device
    
    # Convert to tensors if needed
    if isinstance(K1, np.ndarray):
        K1 = torch.from_numpy(K1).float().to(device)
        K2 = torch.from_numpy(K2).float().to(device)
        R1 = torch.from_numpy(R1).float().to(device)
        t1 = torch.from_numpy(t1).float().to(device)
        R2 = torch.from_numpy(R2).float().to(device)
        t2 = torch.from_numpy(t2).float().to(device)
    
    # Compute fundamental matrix
    t_diff = t2 - t1
    t_skew = torch.tensor([
        [0, -t_diff[2], t_diff[1]],
        [t_diff[2], 0, -t_diff[0]],
        [-t_diff[1], t_diff[0], 0]
    ], device=device, dtype=torch.float32)
    
    E = t_skew @ (R2 @ R1.T)  # Essential matrix
    F = torch.inverse(K2).T @ E @ torch.inverse(K1)  # Fundamental matrix
    
    # Normalize features
    features1_norm = torch.nn.functional.normalize(features1, p=2, dim=0)  # [D, H, W]
    features2_norm = torch.nn.functional.normalize(features2, p=2, dim=0)  # [D, H, W]
    
    correspondences = []
    confidences = []
    
    # Sample points from first image (subsample for efficiency)
    step = max(1, min(H, W) // 64)  # Adaptive subsampling
    y_samples = torch.arange(step//2, H, step, device=device)
    x_samples = torch.arange(step//2, W, step, device=device)
    
    for y1 in y_samples:
        for x1 in x_samples:
            # Point in first image
            p1 = torch.tensor([x1, y1, 1], device=device, dtype=torch.float32)
            
            # Compute epipolar line in second image
            epipolar_line = F @ p1  # [a, b, c] where ax + by + c = 0
            a, b, c = epipolar_line[0], epipolar_line[1], epipolar_line[2]
            
            if abs(b) < 1e-6:  # Nearly vertical line
                continue
                
            # Sample points along epipolar line
            best_corr = -1
            best_x2, best_y2 = -1, -1
            
            x_search = torch.arange(max(0, x1-search_radius*2), 
                                  min(W, x1+search_radius*2+1), device=device)
            
            for x2 in x_search:
                # y coordinate on epipolar line: y = (-ax - c) / b
                y2 = (-a * x2 - c) / b
                y2_int = int(torch.round(y2))
                
                if 0 <= y2_int < H:
                    # Get features at both points
                    feat1 = features1_norm[:, y1, x1]  # [D]
                    feat2 = features2_norm[:, y2_int, x2]  # [D]
                    
                    # Compute correlation
                    corr = torch.dot(feat1, feat2).item()
                    
                    if corr > best_corr:
                        best_corr = corr
                        best_x2, best_y2 = x2.item(), y2_int
            
            # Store correspondence if correlation is good enough
            if best_corr > 0.5:  # Threshold for valid correspondence
                correspondences.append([x1.item(), y1.item(), best_x2, best_y2])
                confidences.append(best_corr)
    
    if len(correspondences) == 0:
        return np.array([]).reshape(0, 4), np.array([])
    
    return np.array(correspondences), np.array(confidences)


def dense_correlation_matching(features1, features2, K1, K2, R1, t1, R2, t2, subsample_step=4):
    """
    Dense correlation-based matching with epipolar constraints.
    
    Args:
        features1, features2: [D, H, W] dense feature maps
        K1, K2: Camera intrinsics
        R1, t1, R2, t2: Camera extrinsics
        subsample_step: Subsampling step for efficiency
    
    Returns:
        pts1, pts2: [N, 2] corresponding points
        confidences: [N] confidence scores
    """
    # First try epipolar-constrained search for more accurate correspondences
    correspondences, confidences = compute_epipolar_correlations(
        features1, features2, K1, K2, R1, t1, R2, t2, search_radius=8
    )
    
    if len(correspondences) > 100:  # If we have enough correspondences, use them
        pts1 = correspondences[:, :2]
        pts2 = correspondences[:, 2:]
        return pts1, pts2, confidences
    
    # Fallback to dense correlation search if epipolar search fails
    print(f"   🔄 Epipolar search found only {len(correspondences)} matches, using dense correlation...")
    
    # Subsample for efficiency
    D, H, W = features1.shape
    y_coords = torch.arange(subsample_step//2, H, subsample_step, device=features1.device)
    x_coords = torch.arange(subsample_step//2, W, subsample_step, device=features1.device)
    
    correspondences = []
    confidences = []
    
    # Normalize features
    features1_norm = torch.nn.functional.normalize(features1, p=2, dim=0)
    features2_norm = torch.nn.functional.normalize(features2, p=2, dim=0)
    
    for y1 in y_coords:
        for x1 in x_coords:
            feat1 = features1_norm[:, y1, x1]  # [D]
            
            # Compute correlation with all pixels in second image (expensive but comprehensive)
            correlations = torch.sum(feat1.unsqueeze(1).unsqueeze(2) * features2_norm, dim=0)  # [H, W]
            
            # Find best match
            max_corr, max_idx = torch.max(correlations.flatten(), dim=0)
            y2, x2 = divmod(max_idx.item(), W)
            
            if max_corr > 0.6:  # Higher threshold for dense search
                correspondences.append([x1.item(), y1.item(), x2, y2])
                confidences.append(max_corr.item())
    
    if len(correspondences) == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), np.array([])
    
    correspondences = np.array(correspondences)
    pts1 = correspondences[:, :2]
    pts2 = correspondences[:, 2:]
    confidences = np.array(confidences)
    
    return pts1, pts2, confidences


def sample_colors_from_images(pts_3d, image1, image2, K1, K2, R1, t1, R2, t2):
    """
    Sample colors for 3D points by projecting to both images and averaging.
    
    Args:
        pts_3d: [N, 3] 3D points
        image1, image2: PIL Images
        K1, K2, R1, t1, R2, t2: Camera parameters
    
    Returns:
        np.ndarray: [N, 3] RGB colors (0-255)
    """
    # Convert images to numpy arrays
    img1_np = np.array(image1)  # [H, W, 3]
    img2_np = np.array(image2)  # [H, W, 3]
    
    h1, w1 = img1_np.shape[:2]
    h2, w2 = img2_np.shape[:2]
    
    # Project 3D points to both images
    pts_3d_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    
    P1 = K1 @ np.hstack([R1, t1.reshape(-1, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(-1, 1)])
    
    proj1_h = (P1 @ pts_3d_h.T).T
    proj2_h = (P2 @ pts_3d_h.T).T
    
    proj1 = proj1_h[:, :2] / proj1_h[:, 2:3]
    proj2 = proj2_h[:, :2] / proj2_h[:, 2:3]
    
    colors = []
    
    for i in range(len(pts_3d)):
        color1 = None
        color2 = None
        
        # Sample from image 1
        x1, y1 = proj1[i]
        if 0 <= x1 < w1 and 0 <= y1 < h1:
            color1 = img1_np[int(y1), int(x1)]
        
        # Sample from image 2
        x2, y2 = proj2[i]
        if 0 <= x2 < w2 and 0 <= y2 < h2:
            color2 = img2_np[int(y2), int(x2)]
        
        # Average available colors
        if color1 is not None and color2 is not None:
            color = (color1.astype(float) + color2.astype(float)) / 2
        elif color1 is not None:
            color = color1.astype(float)
        elif color2 is not None:
            color = color2.astype(float)
        else:
            color = np.array([128, 128, 128], dtype=float)  # Gray fallback
        
        colors.append(color.astype(np.uint8))
    
    return np.array(colors)


def process_image_pair(model, image_pair, colmap_data, args, dtype, pair_idx):
    """
    Process a single image pair to generate point cloud.
    
    Args:
        model: VGGT model
        image_pair: (image1_name, image2_name, shared_count, baseline)
        colmap_data: COLMAP calibration data
        args: Command line arguments
        dtype: Data type for model
        pair_idx: Index of the pair for naming
    
    Returns:
        dict: Results containing point cloud and metadata
    """
    image1_name, image2_name, shared_count, baseline = image_pair
    
    print(f"\n🔄 Processing pair {pair_idx}: {image1_name} - {image2_name}")
    print(f"   Shared points: {shared_count}, Baseline: {baseline:.3f}")
    
    # Load images
    image1_path = os.path.join(args.scene_dir, "images", image1_name)
    image2_path = os.path.join(args.scene_dir, "images", image2_name)
    
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print(f"   ❌ Images not found")
        return None
    
    # Load original images for color sampling
    image1_orig = Image.open(image1_path).convert('RGB')
    image2_orig = Image.open(image2_path).convert('RGB')
    
    # Preprocess images for VGGT
    images, coords_info = load_and_preprocess_images_square([image1_path, image2_path], args.resolution)
    images = images.to(next(model.parameters()).device)
    
    # Get camera parameters from COLMAP
    cam_data1 = colmap_data['images'][image1_name]
    cam_data2 = colmap_data['images'][image2_name]
    
    K1_orig = cam_data1['intrinsic']
    K2_orig = cam_data2['intrinsic']
    extrinsic1 = cam_data1['extrinsic']
    extrinsic2 = cam_data2['extrinsic']
    R1, t1 = extrinsic1[:3, :3], extrinsic1[:3, 3]
    R2, t2 = extrinsic2[:3, :3], extrinsic2[:3, 3]
    
    # Adjust intrinsics for VGGT preprocessing
    orig_size1 = (image1_orig.width, image1_orig.height)
    orig_size2 = (image2_orig.width, image2_orig.height)
    
    K1 = adjust_intrinsics_for_preprocessing(K1_orig, orig_size1, args.resolution)
    K2 = adjust_intrinsics_for_preprocessing(K2_orig, orig_size2, args.resolution)
    
    # Extract dense features
    try:
        features = extract_dense_features(model, images, dtype)
        print(f"   ✅ Extracted features: {features.shape}")
        
    except Exception as e:
        print(f"   ❌ Feature extraction failed: {e}")
        return None
    
    # Match features using dense correlation approach
    try:
        # features should be [B, D, H, W] where B=2, D is feature dimension
        features1 = features[0]  # [D, H, W] - first image features
        features2 = features[1]  # [D, H, W] - second image features
        
        D, H, W = features1.shape
        print(f"   🔍 Dense features shape: {features1.shape}, {features2.shape}")
        print(f"   🔍 Feature dimension: {D}, spatial resolution: {H}x{W}")
        
        # Use dense correlation matching with epipolar constraints
        pts1, pts2, conf = dense_correlation_matching(
            features1, features2, K1, K2, R1, t1, R2, t2, 
            subsample_step=args.subsample_step
        )
        
        print(f"   ✅ Found {len(pts1)} dense correspondences")
        
        # Limit correspondences for efficiency
        if len(pts1) > args.max_correspondences:
            indices = np.argsort(conf)[-args.max_correspondences:]
            pts1 = pts1[indices]
            pts2 = pts2[indices]
            conf = conf[indices]
            print(f"   📉 Limited to {len(pts1)} correspondences")
        
    except Exception as e:
        print(f"   ❌ Dense correlation matching failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    if len(pts1) < 50:  # Higher threshold for dense matching
        print(f"   ❌ Too few correspondences: {len(pts1)}")
        return None
    
    # Triangulate correspondences
    try:
        cam_center1 = -R1.T @ t1
        cam_center2 = -R2.T @ t2
        
        pts_3d = triangulate_correspondences(pts1, pts2, K1, K2, R1, t1, R2, t2)
        
        print(f"   ✅ Triangulated {len(pts_3d)} points")
        
    except Exception as e:
        print(f"   ❌ Triangulation failed: {e}")
        return None
    
    # Filter correspondences
    try:
        valid_mask = filter_correspondences(
            pts1, pts2, pts_3d, K1, K2, R1, t1, R2, t2,
            cam_center1, cam_center2, args
        )
        
        if np.sum(valid_mask) < 10:
            print(f"   ❌ Too few points after filtering: {np.sum(valid_mask)}")
            return None
        
        # Apply filtering
        pts1_filtered = pts1[valid_mask]
        pts2_filtered = pts2[valid_mask]
        pts_3d_filtered = pts_3d[valid_mask]
        
    except Exception as e:
        print(f"   ❌ Filtering failed: {e}")
        return None
    
    # Sample colors
    try:
        colors = sample_colors_from_images(
            pts_3d_filtered, image1_orig, image2_orig,
            K1_orig, K2_orig, R1, t1, R2, t2
        )
        
        print(f"   ✅ Sampled colors for {len(colors)} points")
        
    except Exception as e:
        print(f"   ❌ Color sampling failed: {e}")
        colors = np.full((len(pts_3d_filtered), 3), 128, dtype=np.uint8)
    
    # Create pair output directory
    pair_dir = os.path.join(args.output_dir, f"pair_{pair_idx:03d}")
    os.makedirs(pair_dir, exist_ok=True)
    
    # Save point cloud
    try:
        point_cloud = trimesh.PointCloud(vertices=pts_3d_filtered, colors=colors)
        ply_path = os.path.join(pair_dir, "pointcloud.ply")
        point_cloud.export(ply_path)
        
        print(f"   💾 Saved point cloud: {ply_path}")
        
    except Exception as e:
        print(f"   ❌ Point cloud save failed: {e}")
        return None
    
    # Save pair metadata
    metadata = {
        'pair_index': pair_idx,
        'image1': image1_name,
        'image2': image2_name,
        'shared_3d_points': shared_count,
        'baseline_distance': baseline,
        'num_correspondences_raw': len(pts1),
        'num_correspondences_filtered': len(pts_3d_filtered),
        'camera_parameters': {
            'K1_original': K1_orig.tolist(),
            'K2_original': K2_orig.tolist(),
            'K1_adjusted': K1.tolist(),
            'K2_adjusted': K2.tolist(),
            'R1': R1.tolist(),
            't1': t1.tolist(),
            'R2': R2.tolist(),
            't2': t2.tolist(),
            'camera_center1': cam_center1.tolist(),
            'camera_center2': cam_center2.tolist()
        },
        'processing_parameters': vars(args)
    }
    
    metadata_path = os.path.join(pair_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Cleanup
    del images, features, point_cloud
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'pair_dir': pair_dir,
        'ply_path': ply_path,
        'num_points': len(pts_3d_filtered),
        'metadata': metadata
    }


def merge_all_point_clouds(output_dir, pair_results):
    """
    Merge all pair point clouds into a single dense point cloud.
    
    Args:
        output_dir: Output directory
        pair_results: List of results from process_image_pair
    
    Returns:
        str: Path to merged point cloud
    """
    print(f"\n🔗 Merging {len(pair_results)} point clouds...")
    
    all_vertices = []
    all_colors = []
    total_points = 0
    
    for result in pair_results:
        if result is None:
            continue
        
        try:
            # Load point cloud
            point_cloud = trimesh.load(result['ply_path'])
            
            if hasattr(point_cloud, 'vertices') and len(point_cloud.vertices) > 0:
                vertices = point_cloud.vertices
                colors = point_cloud.colors if hasattr(point_cloud, 'colors') else None
                
                all_vertices.append(vertices)
                if colors is not None:
                    all_colors.append(colors)
                else:
                    # Create default gray colors
                    gray_colors = np.full((len(vertices), 3), 128, dtype=np.uint8)
                    all_colors.append(gray_colors)
                
                total_points += len(vertices)
                print(f"   ✅ {result['pair_dir']}: {len(vertices)} points")
            
        except Exception as e:
            print(f"   ❌ Failed to load {result['ply_path']}: {e}")
    
    if len(all_vertices) == 0:
        print(f"   ❌ No point clouds to merge")
        return None
    
    # Combine all vertices and colors
    combined_vertices = np.vstack(all_vertices)
    combined_colors = np.vstack(all_colors)
    
    # Save merged point cloud
    merged_path = os.path.join(output_dir, "dense_pointcloud.ply")
    merged_pointcloud = trimesh.PointCloud(vertices=combined_vertices, colors=combined_colors)
    merged_pointcloud.export(merged_path)
    
    print(f"   ✅ Merged point cloud saved: {merged_path}")
    print(f"   📊 Total points: {len(combined_vertices):,}")
    print(f"   📐 Point range: X[{combined_vertices[:,0].min():.3f}, {combined_vertices[:,0].max():.3f}], "
          f"Y[{combined_vertices[:,1].min():.3f}, {combined_vertices[:,1].max():.3f}], "
          f"Z[{combined_vertices[:,2].min():.3f}, {combined_vertices[:,2].max():.3f}]")
    
    return merged_path


def save_processing_summary(output_dir, args, good_pairs, pair_results):
    """Save summary of processing results."""
    
    successful_pairs = [r for r in pair_results if r is not None]
    total_points = sum(r['num_points'] for r in successful_pairs)
    
    summary = {
        'processing_parameters': vars(args),
        'statistics': {
            'total_pairs_found': len(good_pairs),
            'pairs_processed': len(pair_results),
            'pairs_successful': len(successful_pairs),
            'total_points_generated': total_points,
            'average_points_per_pair': total_points / len(successful_pairs) if successful_pairs else 0
        },
        'pair_results': [
            {
                'pair_index': i,
                'image1': good_pairs[i][0],
                'image2': good_pairs[i][1],
                'shared_points': good_pairs[i][2],
                'baseline': good_pairs[i][3],
                'success': pair_results[i] is not None,
                'points_generated': pair_results[i]['num_points'] if pair_results[i] else 0
            }
            for i in range(len(pair_results))
        ]
    }
    
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Processing summary saved: {summary_path}")


def main():
    args = parse_args()
    print("🚀 Dense Feature Correspondence and Triangulation")
    print("Arguments:", vars(args))
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load VGGT model with DPT head for dense feature extraction
    print("\n📚 Loading VGGT model...")
    model = VGGT(
        img_size=518,
        enable_camera=False,
        enable_point=False,
        enable_depth=True,  # Use DPT head for dense features
        enable_track=False
    )
    
    # Configure DPT head for feature extraction (feature_only mode)
    if hasattr(model, 'depth_head') and model.depth_head is not None:
        model.depth_head.feature_only = True
        print("✅ Configured DPT head for dense feature extraction")
    
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state_dict = torch.hub.load_state_dict_from_url(_URL)
    
    # Filter state dict (keep depth_head, skip others)
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if not any(skip_key in k for skip_key in ['point_head', 'track_head', 'camera_head'])
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded")
    
    # Load COLMAP calibration
    print(f"\n📷 Loading COLMAP calibration from {args.reference_calibration}...")
    reconstruction = load_reconstruction(args.reference_calibration)
    colmap_data = load_colmap_calibration(args.reference_calibration)
    
    if colmap_data is None:
        raise ValueError(f"Failed to load COLMAP calibration from {args.reference_calibration}")
    
    print(f"✅ Loaded calibration for {len(colmap_data['images'])} images")
    
    # Analyze 3D point sharing
    print(f"\n🔍 Analyzing 3D point sharing...")
    point_sharing_info = analyze_3d_point_sharing_for_pairs(reconstruction)
    
    # Find good image pairs
    good_pairs = find_good_image_pairs(
        point_sharing_info, 
        args.min_shared_points, 
        args.min_baseline_ratio
    )
    
    if len(good_pairs) == 0:
        print("❌ No suitable image pairs found")
        return False
    
    # Limit number of pairs if specified
    if args.max_pairs is not None and len(good_pairs) > args.max_pairs:
        good_pairs = good_pairs[:args.max_pairs]
        print(f"📉 Limited to {len(good_pairs)} pairs")
    
    # Process all pairs
    print(f"\n🔄 Processing {len(good_pairs)} image pairs...")
    pair_results = []
    
    for i, pair in enumerate(good_pairs):
        result = process_image_pair(model, pair, colmap_data, args, dtype, i)
        pair_results.append(result)
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Merge all point clouds
    merged_path = merge_all_point_clouds(args.output_dir, pair_results)
    
    # Save processing summary
    save_processing_summary(args.output_dir, args, good_pairs, pair_results)
    
    # Final summary
    successful_pairs = [r for r in pair_results if r is not None]
    print(f"\n🎉 Processing completed!")
    print(f"📊 Results:")
    print(f"   Pairs processed: {len(pair_results)}")
    print(f"   Successful pairs: {len(successful_pairs)}")
    print(f"   Total points: {sum(r['num_points'] for r in successful_pairs):,}")
    print(f"📁 Output directory: {args.output_dir}")
    if merged_path:
        print(f"🌐 Merged point cloud: {merged_path}")
    
    return True


if __name__ == "__main__":
    main() 