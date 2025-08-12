# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as R


def load_reconstruction(sparse_dir):
    """Load a COLMAP reconstruction from sparse directory."""
    try:
        reconstruction = pycolmap.Reconstruction(sparse_dir)
        return reconstruction
    except Exception as e:
        print(f"Error loading reconstruction from {sparse_dir}: {e}")
        return None


def extract_camera_poses(reconstruction):
    """Extract camera poses (positions and orientations) from reconstruction."""
    poses = {}
    
    for image_id, image in reconstruction.images.items():
        if image.registered:
            # Get camera-to-world transformation
            cam_from_world = image.cam_from_world
            
            # Extract rotation and translation from cam_from_world
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation
            
            # Compute camera center as -R.T @ t
            camera_center = -R.T @ t
            
            poses[image.name] = {
                'position': camera_center,
                'rotation': R,
                'image_id': image_id
            }
    
    return poses


def find_common_images(vggt_poses, colmap_poses):
    """Find images that are present in both reconstructions."""
    vggt_names = set(vggt_poses.keys())
    colmap_names = set(colmap_poses.keys())
    
    common_names = vggt_names & colmap_names
    print(f"Found {len(common_names)} common images out of {len(vggt_names)} VGGT and {len(colmap_names)} COLMAP images")
    
    if len(common_names) < 3:
        raise ValueError("Need at least 3 common images for similarity transform estimation")
    
    return sorted(list(common_names))


def similarity_transform_3d(points_src, points_dst):
    """
    Compute similarity transform (rotation, translation, scale) between two 3D point sets.
    Uses Procrustes analysis (Kabsch algorithm).
    
    Args:
        points_src: Nx3 array of source points
        points_dst: Nx3 array of destination points
    
    Returns:
        dict with rotation (3x3), translation (3,), scale (float), transformed_points (Nx3)
    """
    assert points_src.shape == points_dst.shape
    assert points_src.shape[1] == 3
    
    if points_src.shape[0] < 3:
        raise ValueError("Need at least 3 points for similarity transform")
    
    # Center the points
    centroid_src = np.mean(points_src, axis=0)
    centroid_dst = np.mean(points_dst, axis=0)
    
    points_src_centered = points_src - centroid_src
    points_dst_centered = points_dst - centroid_dst
    
    # Compute scale as ratio of root mean square deviations
    scale_src = np.sqrt(np.mean(np.sum(points_src_centered**2, axis=1)))
    scale_dst = np.sqrt(np.mean(np.sum(points_dst_centered**2, axis=1)))
    
    if scale_src < 1e-10:
        raise ValueError("Source points are too close to centroid")
    
    scale = scale_dst / scale_src
    
    # Normalize points for rotation estimation
    points_src_normalized = points_src_centered / scale_src
    points_dst_normalized = points_dst_centered / scale_dst
    
    # Compute rotation using SVD (Kabsch algorithm)
    H = points_src_normalized.T @ points_dst_normalized
    U, S, Vt = np.linalg.svd(H)
    
    # Ensure proper rotation (det = 1)
    d = np.linalg.det(Vt.T @ U.T)
    if d < 0:
        Vt[-1, :] *= -1
    
    rotation = Vt.T @ U.T
    
    # Compute translation
    translation = centroid_dst - scale * (rotation @ centroid_src)
    
    # Apply transform to source points: T(s*R*p + t) = s*R*p + t
    transformed_points = scale * (rotation @ points_src.T).T + translation
    
    return {
        'rotation': rotation,
        'translation': translation,
        'scale': scale,
        'transformed_points': transformed_points,
        'rmse': np.sqrt(np.mean(np.sum((transformed_points - points_dst)**2, axis=1)))
    }


def robust_similarity_transform_3d(points_src, points_dst, outlier_threshold=2.0, max_iterations=10):
    """
    Compute robust similarity transform that iteratively removes outliers.
    
    Args:
        points_src: Nx3 array of source points
        points_dst: Nx3 array of destination points
        outlier_threshold: Remove points with error > threshold * median_error
        max_iterations: Maximum number of outlier removal iterations
    
    Returns:
        dict with robust transform and inlier information
    """
    assert points_src.shape == points_dst.shape
    assert points_src.shape[1] == 3
    
    if points_src.shape[0] < 3:
        raise ValueError("Need at least 3 points for similarity transform")
    
    # Start with all points
    inlier_mask = np.ones(len(points_src), dtype=bool)
    best_result = None
    best_rmse = float('inf')
    
    for iteration in range(max_iterations):
        # Use current inliers
        current_src = points_src[inlier_mask]
        current_dst = points_dst[inlier_mask]
        
        if np.sum(inlier_mask) < 3:
            print(f"Warning: Only {np.sum(inlier_mask)} inliers remaining, stopping")
            break
        
        # Compute transform
        result = similarity_transform_3d(current_src, current_dst)
        
        # Compute errors for ALL points
        all_transformed = result['scale'] * (result['rotation'] @ points_src.T).T + result['translation']
        errors = np.linalg.norm(all_transformed - points_dst, axis=1)
        
        # Update inliers based on median error threshold
        median_error = np.median(errors[inlier_mask])
        threshold = outlier_threshold * median_error
        
        new_inlier_mask = errors < threshold
        
        # Check convergence
        if np.array_equal(inlier_mask, new_inlier_mask):
            print(f"Converged after {iteration + 1} iterations")
            break
        
        inlier_mask = new_inlier_mask
        
        # Keep track of best result
        if result['rmse'] < best_rmse:
            best_rmse = result['rmse']
            best_result = result.copy()
            best_result['inlier_mask'] = inlier_mask.copy()
            best_result['outlier_count'] = np.sum(~inlier_mask)
            best_result['inlier_count'] = np.sum(inlier_mask)
        
        print(f"Iteration {iteration + 1}: {np.sum(inlier_mask)} inliers, RMSE: {result['rmse']:.6f}")
    
    if best_result is None:
        # Fallback to regular transform
        return similarity_transform_3d(points_src, points_dst)
    
    # Final transform with inliers only
    final_src = points_src[best_result['inlier_mask']]
    final_dst = points_dst[best_result['inlier_mask']]
    final_result = similarity_transform_3d(final_src, final_dst)
    
    # Add robust-specific information
    final_result['inlier_mask'] = best_result['inlier_mask']
    final_result['outlier_count'] = best_result['outlier_count']
    final_result['inlier_count'] = best_result['inlier_count']
    final_result['robust_rmse'] = final_result['rmse']
    
    # Compute RMSE for all points (including outliers)
    all_transformed = final_result['scale'] * (final_result['rotation'] @ points_src.T).T + final_result['translation']
    final_result['all_points_rmse'] = np.sqrt(np.mean(np.sum((all_transformed - points_dst)**2, axis=1)))
    
    return final_result


def compute_similarity_transform(source_sparse_dir, target_sparse_dir, verbose=True, use_robust=False):
    """
    Compute similarity transform between source and target reconstructions.
    Transform source reconstruction to target coordinate system.
    
    Args:
        vggt_sparse_dir: Path to VGGT sparse reconstruction directory
        colmap_sparse_dir: Path to COLMAP sparse reconstruction directory
        verbose: Whether to print progress information
        use_robust: Whether to use robust transform (removes outliers)
    
    Returns:
        dict: Similarity transform parameters and statistics
    """
    if verbose:
        print("Loading reconstructions...")
    
    # Load reconstructions
    source_reconstruction = load_reconstruction(source_sparse_dir)
    if source_reconstruction is None:
        raise ValueError(f"Failed to load VGGT reconstruction from {source_sparse_dir}")
    
    target_reconstruction = load_reconstruction(target_sparse_dir)
    if target_reconstruction is None:
        raise ValueError(f"Failed to load COLMAP reconstruction from {target_sparse_dir}")
    
    if verbose:
        print("Extracting camera poses...")
    
    # Extract poses
    source_poses = extract_camera_poses(source_reconstruction)
    target_poses = extract_camera_poses(target_reconstruction)
    
    if verbose:
        print(f"Source reconstruction: {len(source_poses)} images")
        print(f"Target reconstruction: {len(target_poses)} images")
    
    # Find common images
    common_names = find_common_images(source_poses, target_poses)
    
    # Extract positions for common images
    source_common_positions = np.array([source_poses[name]['position'] for name in common_names])
    target_common_positions = np.array([target_poses[name]['position'] for name in common_names])
    
    if verbose:
        print("Computing similarity transform...")
    
    # Compute similarity transform
    if use_robust:
        transform_result = robust_similarity_transform_3d(source_common_positions, target_common_positions)
        if verbose:
            print(f"Robust Similarity Transform Results:")
            print(f"  Scale: {transform_result['scale']:.6f}")
            print(f"  Translation: [{transform_result['translation'][0]:.6f}, {transform_result['translation'][1]:.6f}, {transform_result['translation'][2]:.6f}]")
            print(f"  Rotation (axis-angle): {R.from_matrix(transform_result['rotation']).as_rotvec()}")
            print(f"  Inliers: {transform_result.get('inlier_count', len(common_names))}/{len(common_names)}")
            print(f"  Outliers removed: {transform_result.get('outlier_count', 0)}")
            print(f"  RMSE (inliers only): {transform_result.get('robust_rmse', transform_result['rmse']):.6f}")
            print(f"  RMSE (all points): {transform_result.get('all_points_rmse', transform_result['rmse']):.6f}")
    else:
        transform_result = similarity_transform_3d(source_common_positions, target_common_positions)
        if verbose:
            print(f"Similarity Transform Results:")
            print(f"  Scale: {transform_result['scale']:.6f}")
            print(f"  Translation: [{transform_result['translation'][0]:.6f}, {transform_result['translation'][1]:.6f}, {transform_result['translation'][2]:.6f}]")
            print(f"  Rotation (axis-angle): {R.from_matrix(transform_result['rotation']).as_rotvec()}")
            print(f"  RMSE after alignment: {transform_result['rmse']:.6f}")
    
    return {
        'rotation': transform_result['rotation'],
        'translation': transform_result['translation'],
        'scale': transform_result['scale'],
        'rmse': transform_result.get('robust_rmse', transform_result['rmse']),
        'all_points_rmse': transform_result.get('all_points_rmse', transform_result['rmse']),
        'num_common_images': len(common_names),
        'common_images': common_names,
        'inlier_count': transform_result.get('inlier_count', len(common_names)),
        'outlier_count': transform_result.get('outlier_count', 0),
        'used_robust': use_robust
    }


def apply_similarity_transform(points, transform):
    """
    Apply similarity transform to a set of 3D points.
    
    Args:
        points: Nx3 array of 3D points
        transform: Transform dict from compute_similarity_transform
    
    Returns:
        Nx3 array of transformed points
    """
    if points.shape[0] == 0:
        return points
    
    # Apply scale, rotation, and translation: T(p) = s*R*p + t
    transformed_points = transform['scale'] * (transform['rotation'] @ points.T).T + transform['translation']
    return transformed_points


def transform_point_cloud_to_colmap_frame(points, colors, transform):
    """
    Transform a point cloud from VGGT coordinate system to COLMAP coordinate system.
    
    Args:
        points: Nx3 array of 3D points in VGGT coordinates
        colors: Nx3 array of RGB colors (0-255)
        transform: Transform dict from compute_similarity_transform
    
    Returns:
        tuple: (transformed_points, colors) - colors are unchanged
    """
    transformed_points = apply_similarity_transform(points, transform)
    return transformed_points, colors 