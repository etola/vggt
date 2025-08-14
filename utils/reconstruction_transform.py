import os
import numpy as np
import pycolmap
from typing import Dict, List, Tuple


def load_reconstruction(sparse_dir: str) -> pycolmap.Reconstruction:
    """Load a COLMAP reconstruction from a directory."""
    if not os.path.isdir(sparse_dir):
        raise FileNotFoundError(f"Reconstruction directory not found: {sparse_dir}")
    try:
        return pycolmap.Reconstruction(sparse_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to load COLMAP reconstruction from {sparse_dir}: {exc}")


def extract_camera_centers_and_rotations(reconstruction: pycolmap.Reconstruction) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract camera centers and rotations for all registered images keyed by image name.

    Returns a dict: name -> {"center": (3,), "rotation": (3,3), "image_id": int, "camera_id": int}
    """
    poses: Dict[str, Dict[str, np.ndarray]] = {}
    for image_id, image in reconstruction.images.items():
        if not image.registered:
            continue
        cam_from_world = image.cam_from_world
        rotation_matrix: np.ndarray = cam_from_world.rotation.matrix()
        translation_vec: np.ndarray = cam_from_world.translation
        # Camera center in world coordinates: C = -R^T t
        camera_center = -rotation_matrix.T @ translation_vec
        poses[image.name] = {
            "center": camera_center,
            "rotation": rotation_matrix,
            "image_id": image_id,
            "camera_id": image.camera_id,
        }
    return poses


def match_common_image_names(source_poses: Dict[str, Dict], target_poses: Dict[str, Dict]) -> List[str]:
    """Return sorted list of common image names present in both reconstructions."""
    common = sorted(list(set(source_poses.keys()) & set(target_poses.keys())))
    if len(common) < 2:
        raise ValueError("At least 2 common images are required to estimate the scale.")
    return common


def _pairwise_distance_ratios(points_src: np.ndarray, points_dst: np.ndarray) -> np.ndarray:
    """Compute ratios of pairwise distances ||p_i - p_j||_dst / ||p_i - p_j||_src for all i<j."""
    n = points_src.shape[0]
    if n < 2:
        raise ValueError("Need at least two points to compute pairwise distance ratios.")
    ratios: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            ds = np.linalg.norm(points_src[i] - points_src[j])
            dt = np.linalg.norm(points_dst[i] - points_dst[j])
            if ds <= 1e-12:
                continue
            ratios.append(dt / ds)
    if len(ratios) == 0:
        raise ValueError("Degenerate configuration: zero pairwise distances in source set.")
    return np.asarray(ratios)


def estimate_scale_from_centers(points_src: np.ndarray, points_dst: np.ndarray, robust: bool = True) -> float:
    """Estimate scale s such that s * points_src ~ points_dst (both in same frame up to rigid).

    - robust=True uses the median of pairwise distance ratios, works with as few as 2 points.
    - robust=False uses RMS deviation about centroid ratio (Umeyama-style), requires >= 2 points.
    """
    if points_src.shape[0] < 2 or points_dst.shape[0] < 2:
        raise ValueError("Need at least two points in each set to estimate scale.")

    if robust:
        ratios = _pairwise_distance_ratios(points_src, points_dst)
        return float(np.median(ratios))
    else:
        src_centroid = np.mean(points_src, axis=0)
        dst_centroid = np.mean(points_dst, axis=0)
        src_rms = np.sqrt(np.mean(np.sum((points_src - src_centroid) ** 2, axis=1)))
        dst_rms = np.sqrt(np.mean(np.sum((points_dst - dst_centroid) ** 2, axis=1)))
        if src_rms <= 1e-12:
            raise ValueError("Degenerate configuration: source points too concentrated around centroid.")
        return float(dst_rms / src_rms)


def estimate_rigid_transform(points_src: np.ndarray, points_dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate rotation R (3x3) and translation t (3,) such that R * p + t ≈ q.

    Supports N >= 2 points. For N == 2 the solution chooses the minimal rotation that aligns the segment.
    """
    assert points_src.shape == points_dst.shape
    n = points_src.shape[0]
    if n < 2:
        raise ValueError("Need at least two corresponding points to estimate rigid transform.")

    mu_src = np.mean(points_src, axis=0)
    mu_dst = np.mean(points_dst, axis=0)
    X = points_src - mu_src
    Y = points_dst - mu_dst

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R_est = Vt.T @ U.T

    # Ensure a proper rotation (determinant +1)
    if np.linalg.det(R_est) < 0:
        Vt[-1, :] *= -1
        R_est = Vt.T @ U.T

    t_est = mu_dst - R_est @ mu_src
    return R_est, t_est


def estimate_similarity_transform_from_recons(
    source_sparse_dir: str,
    target_sparse_dir: str,
    robust_scale: bool = True,
) -> Dict:
    """Estimate similarity transform (s, R, t) aligning source to target using matched image names and camera centers.

    Steps:
      1) Load reconstructions; extract camera centers; match on image names
      2) Estimate scale s from centers
      3) Scale source centers: s * C_src
      4) Estimate R, t via Kabsch on scaled centers
    Returns a dict with keys: 'scale', 'rotation', 'translation', 'rmse', 'num_common', 'common_images'
    """
    source_rec = load_reconstruction(source_sparse_dir)
    target_rec = load_reconstruction(target_sparse_dir)

    src_poses = extract_camera_centers_and_rotations(source_rec)
    dst_poses = extract_camera_centers_and_rotations(target_rec)

    common = match_common_image_names(src_poses, dst_poses)

    src_centers = np.asarray([src_poses[name]["center"] for name in common], dtype=float)
    dst_centers = np.asarray([dst_poses[name]["center"] for name in common], dtype=float)

    scale = estimate_scale_from_centers(src_centers, dst_centers, robust=robust_scale)

    scaled_src = scale * src_centers

    # Align the source's first camera to the destination's first camera using their centers and rotation matrices
    # Use the scaled source center and its rotation, and the destination center and rotation
    if len(common) == 0:
        raise ValueError("No common images found between source and target reconstructions.")

    # Use the first common image
    first_img = common[0]
    src_pose = src_poses[first_img]
    dst_pose = dst_poses[first_img]

    # Scaled source center
    src_center_scaled = scale * np.asarray(src_pose["center"], dtype=float)
    dst_center = np.asarray(dst_pose["center"], dtype=float)

    # Rotation matrices
    src_rot = np.asarray(src_pose["rotation"], dtype=float)
    dst_rot = np.asarray(dst_pose["rotation"], dtype=float)

    # The rotation that aligns the source camera to the destination camera
    # R_est * src_rot = dst_rot  =>  R_est = dst_rot @ src_rot.T
    R_est = dst_rot @ src_rot.T

    # The translation that aligns the (rotated) scaled source center to the destination center
    t_est = dst_center - R_est @ src_center_scaled


    # R_est, t_est = estimate_rigid_transform(scaled_src, dst_centers)

    transformed = (R_est @ scaled_src.T).T + t_est
    rmse = float(np.sqrt(np.mean(np.sum((transformed - dst_centers) ** 2, axis=1))))

    # Validate transform by applying to original centers using point transform function
    validated_centers = np.array([
        apply_similarity_transform_to_point(center, scale, R_est, t_est) 
        for center in src_centers
    ])
    
    # For validation: apply the estimated transform to the camera R, t and measure the error
    # between the transformed camera's z axis and the target camera's z axis

    # We'll compute the angle (in degrees) between the transformed source camera z-axis and the target camera z-axis
    z_axis_errors = []
    for name in common:
        src_pose = src_poses[name]
        dst_pose = dst_poses[name]

        # Source camera center and rotation
        src_center = np.asarray(src_pose["center"], dtype=float)
        src_rot = np.asarray(src_pose["rotation"], dtype=float)

        # Target camera rotation
        dst_rot = np.asarray(dst_pose["rotation"], dtype=float)

        # Transform the source camera center and rotation
        src_center_trans = apply_similarity_transform_to_point(src_center, scale, R_est, t_est)
        src_rot_trans = R_est @ src_rot

        # Camera z-axis in world coordinates is R^T @ [0, 0, 1]
        src_z_axis = src_rot_trans.T @ np.array([0, 0, 1])
        dst_z_axis = dst_rot.T @ np.array([0, 0, 1])

        # Normalize
        src_z_axis /= np.linalg.norm(src_z_axis)
        dst_z_axis /= np.linalg.norm(dst_z_axis)

        # Compute angle between z-axes
        dot = np.clip(np.dot(src_z_axis, dst_z_axis), -1.0, 1.0)
        angle_deg = np.arccos(dot) * 180.0 / np.pi
        z_axis_errors.append(angle_deg)

    z_axis_errors = np.array(z_axis_errors)
    max_z_axis_error = float(np.max(z_axis_errors))
    mean_z_axis_error = float(np.mean(z_axis_errors))
    rms_z_axis_error = float(np.sqrt(np.mean(z_axis_errors ** 2)))



    # Check consistency between batch and individual transforms
    validation_rmse = float(np.sqrt(np.mean(np.sum((validated_centers - dst_centers) ** 2, axis=1))))
    transform_consistency_error = float(np.sqrt(np.mean(np.sum((validated_centers - transformed) ** 2, axis=1))))
    
    if transform_consistency_error > 1e-12:
        raise RuntimeError(f"Transform validation failed: consistency error {transform_consistency_error}")

    return {
        "scale": scale,
        "rotation": R_est,
        "translation": t_est,
        "rmse": rmse,
        "validation_rmse": validation_rmse,
        "num_common": len(common),
        "common_images": common,
    }


def apply_similarity_transform_to_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    only_image_names: List[str] = None,
) -> pycolmap.Reconstruction:
    """Apply similarity transform to a COLMAP reconstruction and return a new reconstruction object.

    The intrinsics are copied as-is; only registered images are transformed. If only_image_names is given,
    only those images (by name) are included. 3D points are also transformed using the same similarity transform.
    """
    new_rec = pycolmap.Reconstruction()
    # Copy cameras first
    for cam_id, camera in reconstruction.cameras.items():
        new_rec.add_camera(camera)

    # Transform 3D points using similarity transform: P' = s * R * P + t
    for point3d_id, point3d in reconstruction.points3D.items():
        original_xyz = point3d.xyz
        transformed_xyz = scale * (rotation @ original_xyz) + translation
        
        new_point3d = pycolmap.Point3D(
            id=point3d_id,
            xyz=transformed_xyz,
            color=point3d.color,
            error=point3d.error,
            track=point3d.track
        )
        new_rec.add_point3D(new_point3d)

    # Select images
    images_to_use: List[Tuple[int, pycolmap.Image]] = []
    for img_id, img in reconstruction.images.items():
        if not img.registered:
            continue
        if only_image_names is not None and img.name not in only_image_names:
            continue
        images_to_use.append((img_id, img))

    for img_id, img in images_to_use:
        cam_from_world = img.cam_from_world
        R_orig = cam_from_world.rotation.matrix()
        t_orig = cam_from_world.translation
        center = -R_orig.T @ t_orig

        # Apply similarity: C' = s * R * C + t, R' = R * R_orig
        new_center = scale * (rotation @ center) + translation
        new_R = rotation @ R_orig
        new_t = -new_R @ new_center

        new_img = pycolmap.Image(
            id=img_id,
            name=img.name,
            camera_id=img.camera_id,
            cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(new_R), new_t),
        )
        new_img.points2D = img.points2D  # Copy 2D points and their associations
        new_img.registered = True
        new_rec.add_image(new_img)

    return new_rec


def apply_similarity_transform_to_point(
    point: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply similarity transform to a single 3D point.
    
    Args:
        point: 3D point as (3,) array
        scale: Scale factor
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
        
    Returns:
        Transformed 3D point as (3,) array: T(p) = s * R * p + t
    """
    point = np.asarray(point)
    if point.shape != (3,):
        raise ValueError(f"Point must be a 3D vector, got shape {point.shape}")
    
    return scale * (rotation @ point) + translation


def apply_similarity_transform(points: np.ndarray, transform: Dict) -> np.ndarray:
    """Apply similarity transform to a set of 3D points.
    
    Args:
        points: Nx3 array of 3D points
        transform: Transform dict with 'scale', 'rotation', 'translation' keys
        
    Returns:
        Nx3 array of transformed points
    """
    if points.shape[0] == 0:
        return points
    
    # Apply scale, rotation, and translation: T(p) = s*R*p + t
    transformed_points = transform['scale'] * (transform['rotation'] @ points.T).T + transform['translation']
    return transformed_points


def transform_point_cloud_to_colmap_frame(points: np.ndarray, colors: np.ndarray, transform: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a point cloud using similarity transform.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (0-255)
        transform: Transform dict with 'scale', 'rotation', 'translation' keys
        
    Returns:
        tuple: (transformed_points, colors) - colors are unchanged
    """
    transformed_points = apply_similarity_transform(points, transform)
    return transformed_points, colors


def compute_similarity_transform(source_sparse_dir: str, target_sparse_dir: str, verbose: bool = True, use_robust: bool = False) -> Dict:
    """Compute similarity transform between source and target reconstructions.
    
    Compatibility wrapper for the original compute_similarity_transform function.
    
    Args:
        source_sparse_dir: Path to source sparse reconstruction directory
        target_sparse_dir: Path to target sparse reconstruction directory  
        verbose: Whether to print progress information
        use_robust: Whether to use robust transform (currently maps to robust_scale)
        
    Returns:
        dict: Similarity transform parameters and statistics
    """
    return estimate_similarity_transform_from_recons(
        source_sparse_dir=source_sparse_dir,
        target_sparse_dir=target_sparse_dir,
        robust_scale=use_robust
    )


def estimate_scale_only_from_recons(
    source_sparse_dir: str,
    target_sparse_dir: str,
    robust_scale: bool = True,
) -> Dict:
    """Estimate only the scale factor aligning source to target using matched image names and camera centers.
    
    This is a simplified version of estimate_similarity_transform_from_recons that only computes
    the scale component, which is more efficient when only scale is needed (e.g., for depth map scaling).

    Steps:
      1) Load reconstructions; extract camera centers; match on image names
      2) Estimate scale s from centers using robust or least-squares method
      
    Args:
        source_sparse_dir: Path to source COLMAP reconstruction directory
        target_sparse_dir: Path to target COLMAP reconstruction directory  
        robust_scale: Whether to use robust scale estimation (median of pairwise ratios)
        
    Returns:
        Dict with keys: 'scale', 'rmse_estimate', 'num_common', 'common_images'
    """
    source_rec = load_reconstruction(source_sparse_dir)
    target_rec = load_reconstruction(target_sparse_dir)

    src_poses = extract_camera_centers_and_rotations(source_rec)
    dst_poses = extract_camera_centers_and_rotations(target_rec)

    common = match_common_image_names(src_poses, dst_poses)

    src_centers = np.asarray([src_poses[name]["center"] for name in common], dtype=float)
    dst_centers = np.asarray([dst_poses[name]["center"] for name in common], dtype=float)

    scale = estimate_scale_from_centers(src_centers, dst_centers, robust=robust_scale)
    
    # Compute a simple scale-only RMSE for validation
    # This applies only scale transformation: scaled_src = scale * src_centers
    # Then computes RMS distance to dst_centers (won't be perfect since no rotation/translation)
    scaled_src = scale * src_centers
    scale_only_rmse = float(np.sqrt(np.mean(np.sum((scaled_src - dst_centers) ** 2, axis=1))))

    return {
        "scale": scale,
        "rmse_estimate": scale_only_rmse,
        "num_common": len(common),
        "common_images": common,
    }


def estimate_scale_only_from_cached_data(
    source_poses: Dict,
    target_poses: Dict,
    robust_scale: bool = True,
) -> Dict:
    """Estimate only the scale factor aligning source to target using pre-loaded camera poses.
    
    This is an optimized version of estimate_scale_only_from_recons that uses pre-loaded
    camera pose data instead of loading reconstructions from disk.
    
    Args:
        source_poses: Camera poses from extract_camera_centers_and_rotations for source
        target_poses: Camera poses from extract_camera_centers_and_rotations for target  
        robust_scale: Whether to use robust scale estimation (median of pairwise ratios)
        
    Returns:
        Dict with keys: 'scale', 'rmse_estimate', 'num_common', 'common_images'
    """
    common = match_common_image_names(source_poses, target_poses)

    src_centers = np.asarray([source_poses[name]["center"] for name in common], dtype=float)
    dst_centers = np.asarray([target_poses[name]["center"] for name in common], dtype=float)

    scale = estimate_scale_from_centers(src_centers, dst_centers, robust=robust_scale)
    
    # Compute a simple scale-only RMSE for validation
    # This applies only scale transformation: scaled_src = scale * src_centers
    # Then computes RMS distance to dst_centers (won't be perfect since no rotation/translation)
    scaled_src = scale * src_centers
    scale_only_rmse = float(np.sqrt(np.mean(np.sum((scaled_src - dst_centers) ** 2, axis=1))))

    return {
        "scale": scale,
        "rmse_estimate": scale_only_rmse,
        "num_common": len(common),
        "common_images": common,
    }


def extract_camera_poses(reconstruction: pycolmap.Reconstruction) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract camera poses (positions and orientations) from reconstruction.
    
    Compatibility wrapper for extract_camera_centers_and_rotations.
    
    Returns:
        Dict mapping image names to pose info with 'position', 'rotation', 'image_id' keys
    """
    poses = extract_camera_centers_and_rotations(reconstruction)
    # Convert to expected format (rename 'center' to 'position')
    for name, pose_info in poses.items():
        pose_info['position'] = pose_info.pop('center')
    return poses


def save_reconstruction_text(reconstruction: pycolmap.Reconstruction, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    reconstruction.write_text(output_dir) 