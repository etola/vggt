#!/usr/bin/env python3
"""
COLMAP Similarity Transform Tool

A command-line utility to compute similarity transforms between two COLMAP calibrations.
Can be used to align reconstructions, validate calibrations, or transform coordinate systems.

Usage:
    python colmap_similarity_transform.py --source /path/to/source/sparse --target /path/to/target/sparse --output /path/to/output
    
Features:
    - Computes similarity transform between two COLMAP reconstructions
    - Supports both regular and robust transforms (outlier removal)
    - Saves transformed calibration in COLMAP format
    - Provides detailed statistics and error analysis
    - Exports results to JSON for further analysis
"""

import numpy as np
import os
import json
import argparse
import pycolmap
import trimesh
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

# Import functions from utils/similarity_transform.py
from utils.similarity_transform import (
    compute_similarity_transform, 
    load_reconstruction, 
    extract_camera_poses,
    apply_similarity_transform
)

# Import VGGT utilities for point cloud generation
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute similarity transform between two COLMAP calibrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python colmap_similarity_transform.py --source colmap1/sparse/0 --target colmap2/sparse/0 --output aligned_colmap1

  # Use robust transform to handle outliers
  python colmap_similarity_transform.py --source src --target dst --output out --robust

  # Transform existing point cloud
  python colmap_similarity_transform.py --source vggt/colmap --target reference/colmap --output aligned --pointcloud input.ply

  # Generate point cloud from depth maps with colors
  python colmap_similarity_transform.py --source vggt/colmap --target reference/colmap --output aligned --depth-maps depth/ --confidence-maps conf/ --images images/

  # Generate test data with known transform
  python colmap_similarity_transform.py --source colmap/sparse/0 --test-mode --output test_transformed
        """
    )
    
    parser.add_argument("--source", type=str, required=True,
                       help="Path to source COLMAP sparse reconstruction directory")
    parser.add_argument("--target", type=str, required=False,
                       help="Path to target COLMAP sparse reconstruction directory")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output directory for transformed calibration")
    parser.add_argument("--pointcloud", type=str, required=False,
                       help="Path to point cloud file (.ply) to transform and save as aligned.ply")
    parser.add_argument("--depth-maps", type=str, required=False,
                       help="Directory containing raw depth maps (.npy files)")
    parser.add_argument("--confidence-maps", type=str, required=False,
                       help="Directory containing confidence maps (.npy files)")
    parser.add_argument("--images", type=str, required=False,
                       help="Directory containing original images for color sampling")
    parser.add_argument("--conf-threshold", type=float, default=2.0,
                       help="Confidence threshold for filtering points (default: 2.0)")
    
    # Transform options
    parser.add_argument("--robust", action="store_true", default=False,
                       help="Use robust similarity transform (removes outliers)")
    parser.add_argument("--outlier-threshold", type=float, default=2.0,
                       help="Outlier threshold for robust transform (default: 2.0)")
    
    # Test mode
    parser.add_argument("--test-mode", action="store_true", default=False,
                       help="Generate test data with known transform (for validation)")
    parser.add_argument("--test-scale", type=float, default=2.5,
                       help="Scale factor for test transform (default: 2.5)")
    parser.add_argument("--test-rotation", type=float, nargs=3, default=[0.3, 0.7, 0.2],
                       help="Rotation axis-angle for test transform (default: [0.3, 0.7, 0.2])")
    parser.add_argument("--test-translation", type=float, nargs=3, default=[10.0, -5.0, 2.0],
                       help="Translation for test transform (default: [10.0, -5.0, 2.0])")
    parser.add_argument("--test-cameras", type=int, default=16,
                       help="Number of cameras to use in test mode (default: 16)")
    
    # Output options
    parser.add_argument("--save-stats", action="store_true", default=True,
                       help="Save detailed statistics to JSON (default: True)")
    parser.add_argument("--quiet", action="store_true", default=False,
                       help="Suppress verbose output")
    
    return parser.parse_args()


def apply_transform_to_reconstruction(source_reconstruction, transform_params, common_image_names=None):
    """
    Apply a similarity transform to a COLMAP reconstruction.
    
    Args:
        source_reconstruction: pycolmap.Reconstruction object to transform
        transform_params: Dict with 'scale', 'rotation', 'translation' keys
        common_image_names: List of image names to transform (None = all registered images)
    
    Returns:
        new_reconstruction: Transformed reconstruction
        transformed_image_names: List of image names that were transformed
    """
    new_reconstruction = pycolmap.Reconstruction()
    
    # Copy cameras (intrinsics don't change)
    for camera_id, camera in source_reconstruction.cameras.items():
        new_reconstruction.add_camera(camera)
    
    # Determine which images to transform
    if common_image_names is None:
        # Transform all registered images
        images_to_transform = [(img_id, img) for img_id, img in source_reconstruction.images.items() if img.registered]
    else:
        # Transform only specified images
        images_to_transform = []
        for img_id, img in source_reconstruction.images.items():
            if img.registered and img.name in common_image_names:
                images_to_transform.append((img_id, img))
    
    images_to_transform.sort(key=lambda x: x[1].name)  # Sort by name for consistency
    transformed_image_names = [img.name for _, img in images_to_transform]
    
    print(f"Transforming {len(images_to_transform)} cameras: {transformed_image_names[:5]}{'...' if len(transformed_image_names) > 5 else ''}")
    
    # Extract transform parameters
    scale = transform_params['scale']
    rotation_matrix = transform_params['rotation']
    translation = transform_params['translation']
    
    for image_id, original_image in images_to_transform:
        # Get original camera center (position in world coordinates)
        cam_from_world = original_image.cam_from_world
        R_orig = cam_from_world.rotation.matrix()
        t_orig = cam_from_world.translation
        camera_center = -R_orig.T @ t_orig
        
        # Apply similarity transform to camera center: T(p) = s*R*p + t
        transformed_center = scale * (rotation_matrix @ camera_center) + translation
        
        # Apply rotation to camera orientation: R_new = R * R_orig
        transformed_rotation = rotation_matrix @ R_orig
        
        # Convert back to cam_from_world format
        # If camera center is C_new and rotation is R_new, then:
        # cam_from_world = [R_new | -R_new * C_new]
        transformed_t = -transformed_rotation @ transformed_center
        
        # Create new pose
        new_cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(transformed_rotation), 
            transformed_t
        )
        
        # Create new image
        new_image = pycolmap.Image(
            id=image_id,
            name=original_image.name,
            camera_id=original_image.camera_id,
            cam_from_world=new_cam_from_world
        )
        new_image.points2D = pycolmap.ListPoint2D([])
        new_image.registered = True
        
        new_reconstruction.add_image(new_image)
    
    return new_reconstruction, transformed_image_names


def save_reconstruction(reconstruction, output_dir):
    """Save reconstruction to directory in ASCII format."""
    os.makedirs(output_dir, exist_ok=True)
    reconstruction.write_text(output_dir)
    print(f"Saved reconstruction to {output_dir}")


def transform_and_save_pointcloud(pointcloud_path, transform_params, output_dir):
    """
    Transform a point cloud using the similarity transform and save it.
    
    Args:
        pointcloud_path: Path to the input point cloud file (.ply)
        transform_params: Dict with 'scale', 'rotation', 'translation' keys
        output_dir: Directory to save the transformed point cloud
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the point cloud
        mesh = trimesh.load(pointcloud_path)
        if not hasattr(mesh, 'vertices'):
            print(f"Error: {pointcloud_path} does not contain vertices")
            return False
        
        points = mesh.vertices
        colors = mesh.colors if hasattr(mesh, 'colors') else None
        
        print(f"Loaded point cloud with {len(points)} points from {pointcloud_path}")
        
        # Apply the similarity transform
        transformed_points = apply_similarity_transform(points, transform_params)
        
        # Create new point cloud with transformed points
        if colors is not None:
            transformed_mesh = trimesh.PointCloud(vertices=transformed_points, colors=colors)
        else:
            transformed_mesh = trimesh.PointCloud(vertices=transformed_points)
        
        # Save the transformed point cloud
        output_path = os.path.join(output_dir, "aligned.ply")
        transformed_mesh.export(output_path)
        
        print(f"Transformed point cloud saved to {output_path}")
        print(f"  Original point range: X[{points[:,0].min():.3f}, {points[:,0].max():.3f}], "
              f"Y[{points[:,1].min():.3f}, {points[:,1].max():.3f}], "
              f"Z[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
        print(f"  Transformed point range: X[{transformed_points[:,0].min():.3f}, {transformed_points[:,0].max():.3f}], "
              f"Y[{transformed_points[:,1].min():.3f}, {transformed_points[:,1].max():.3f}], "
              f"Z[{transformed_points[:,2].min():.3f}, {transformed_points[:,2].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"Error transforming point cloud: {e}")
        return False


def generate_pointcloud_from_depth_maps(depth_maps_dir, confidence_maps_dir, original_calibration_dir, 
                                       output_dir, conf_threshold=2.0, vggt_model_resolution=518, images_dir=None, 
                                       similarity_transform=None):
    """
    Generate point cloud from depth maps using transformed calibration data.
    
    Args:
        depth_maps_dir: Directory containing depth maps (.npy files)
        confidence_maps_dir: Directory containing confidence maps (.npy files)
        transformed_calibration_dir: Directory containing transformed COLMAP calibration
        output_dir: Directory to save the generated point cloud
        conf_threshold: Confidence threshold for filtering points
        vggt_model_resolution: Resolution used by VGGT model (default: 518)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from utils.colmap_utils import load_colmap_calibration
        
        # Load original calibration (before transformation)
        calibration_data = load_colmap_calibration(original_calibration_dir)
        if not calibration_data or 'images' not in calibration_data:
            print(f"Error: Could not load calibration from {original_calibration_dir}")
            return False
        
        # Get list of depth and confidence map files
        depth_files = []
        conf_files = []
        
        if os.path.exists(depth_maps_dir):
            depth_files = sorted([f for f in os.listdir(depth_maps_dir) if f.endswith('_depth.npy')])
        
        if os.path.exists(confidence_maps_dir):
            conf_files = sorted([f for f in os.listdir(confidence_maps_dir) if f.endswith('_confidence.npy')])
        
        if not depth_files or not conf_files:
            print(f"Error: No depth or confidence map files found")
            print(f"  Depth maps dir: {depth_maps_dir}")
            print(f"  Confidence maps dir: {confidence_maps_dir}")
            return False
        
        print(f"Found {len(depth_files)} depth maps and {len(conf_files)} confidence maps")
        
        # Collect all points and colors
        all_points = []
        all_colors = []
        
        for depth_file in depth_files:
            # Extract image name from depth file (e.g., "00000_depth.npy" -> "00000.jpg")
            base_name = depth_file.replace('_depth.npy', '')
            image_name = f"{base_name}.jpg"
            
            # Find corresponding confidence file
            conf_file = f"{base_name}_confidence.npy"
            
            if image_name not in calibration_data['images']:
                print(f"Warning: No calibration data for {image_name}, skipping")
                continue
            
            if conf_file not in conf_files:
                print(f"Warning: No confidence map for {image_name}, skipping")
                continue
            
            # Load depth and confidence maps
            depth_path = os.path.join(depth_maps_dir, depth_file)
            conf_path = os.path.join(confidence_maps_dir, conf_file)
            
            depth_map = np.load(depth_path)
            confidence_map = np.load(conf_path)
            
            # Get camera parameters
            image_data = calibration_data['images'][image_name]
            extrinsic = image_data['extrinsic']
            intrinsic = image_data['intrinsic']
            
            # Filter by confidence
            conf_mask = confidence_map > conf_threshold
            
            if not conf_mask.any():
                print(f"Warning: No points above confidence threshold for {image_name}")
                continue
            
            # Unproject depth map to 3D points
            # Add batch dimension and ensure correct shape (H, W, 1)
            depth_map_batch = depth_map[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
            extrinsic_batch = extrinsic[np.newaxis, ...]  # Add batch dimension
            intrinsic_batch = intrinsic[np.newaxis, ...]  # Add batch dimension
            
            points_3d_batch = unproject_depth_map_to_point_map(
                depth_map_batch, extrinsic_batch, intrinsic_batch
            )
            points_3d = points_3d_batch[0]  # Remove batch dimension
            
            # Apply confidence filter
            filtered_points = points_3d[conf_mask]
            
            # Load and sample colors from original image
            if images_dir and os.path.exists(images_dir):
                # Try to find the original image
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                original_image_path = None
                
                for ext in image_extensions:
                    potential_path = os.path.join(images_dir, f"{base_name}{ext}")
                    if os.path.exists(potential_path):
                        original_image_path = potential_path
                        break
                
                if original_image_path:
                    try:
                        # Load and preprocess image for color sampling
                        image = Image.open(original_image_path).convert('RGB')
                        
                        # Resize image to match depth map resolution
                        depth_h, depth_w = depth_map.shape
                        image_resized = image.resize((depth_w, depth_h), Image.Resampling.LANCZOS)
                        image_array = np.array(image_resized)  # HxWx3, uint8
                        
                        # Sample colors using the same confidence mask
                        filtered_colors = image_array[conf_mask]
                        
                        print(f"  {image_name}: {len(filtered_points)} points (with colors from image)")
                    except Exception as e:
                        print(f"Warning: Could not load colors from {original_image_path}: {e}")
                        # Fallback to gray colors
                        filtered_colors = np.zeros((len(filtered_points), 3), dtype=np.uint8)
                        filtered_colors[:, :] = 128
                else:
                    print(f"Warning: No original image found for {base_name}, using gray colors")
                    filtered_colors = np.zeros((len(filtered_points), 3), dtype=np.uint8)
                    filtered_colors[:, :] = 128
            else:
                # No images directory provided, use gray colors
                filtered_colors = np.zeros((len(filtered_points), 3), dtype=np.uint8)
                filtered_colors[:, :] = 128
            
            all_points.append(filtered_points)
            all_colors.append(filtered_colors)
            
            print(f"  {image_name}: {len(filtered_points)} points")
        
        if not all_points:
            print("Error: No valid points generated from any depth map")
            return False
        
        # Combine all points
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        print(f"Generated {len(combined_points)} total points from depth maps")
        
        # Apply similarity transform if provided
        if similarity_transform is not None:
            print(f"Applying similarity transform to generated point cloud...")
            transformed_points = apply_similarity_transform(combined_points, similarity_transform)
            
            # Create transformed point cloud
            transformed_point_cloud = trimesh.PointCloud(vertices=transformed_points, colors=combined_colors)
            output_path = os.path.join(output_dir, "generated_from_depth.ply")
            transformed_point_cloud.export(output_path)
            
            print(f"Transformed point cloud generated from depth maps saved to {output_path}")
            print(f"  Original point range: X[{combined_points[:,0].min():.3f}, {combined_points[:,0].max():.3f}], "
                  f"Y[{combined_points[:,1].min():.3f}, {combined_points[:,1].max():.3f}], "
                  f"Z[{combined_points[:,2].min():.3f}, {combined_points[:,2].max():.3f}]")
            print(f"  Transformed point range: X[{transformed_points[:,0].min():.3f}, {transformed_points[:,0].max():.3f}], "
                  f"Y[{transformed_points[:,1].min():.3f}, {transformed_points[:,1].max():.3f}], "
                  f"Z[{transformed_points[:,2].min():.3f}, {transformed_points[:,2].max():.3f}]")
        else:
            # Create and save point cloud without transformation
            point_cloud = trimesh.PointCloud(vertices=combined_points, colors=combined_colors)
            output_path = os.path.join(output_dir, "generated_from_depth.ply")
            point_cloud.export(output_path)
            
            print(f"Point cloud generated from depth maps saved to {output_path}")
            print(f"  Point range: X[{combined_points[:,0].min():.3f}, {combined_points[:,0].max():.3f}], "
                  f"Y[{combined_points[:,1].min():.3f}, {combined_points[:,1].max():.3f}], "
                  f"Z[{combined_points[:,2].min():.3f}, {combined_points[:,2].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"Error generating point cloud from depth maps: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_data(source_dir, output_dir, scale, rotation_axis_angle, translation, num_cameras):
    """Create test data with known transformation for validation."""
    print(f"=== Creating Test Data ===")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Scale: {scale}")
    print(f"Rotation (axis-angle): {rotation_axis_angle}")
    print(f"Translation: {translation}")
    print(f"Cameras: {num_cameras}")
    
    # Load source reconstruction
    source_reconstruction = load_reconstruction(source_dir)
    if source_reconstruction is None:
        raise ValueError(f"Failed to load reconstruction from {source_dir}")
    
    # Create rotation matrix
    rotation_matrix = R.from_rotvec(rotation_axis_angle).as_matrix()
    
    # Get first N cameras
    registered_images = [(img_id, img) for img_id, img in source_reconstruction.images.items() if img.registered]
    registered_images.sort(key=lambda x: x[1].name)
    
    if len(registered_images) < num_cameras:
        raise ValueError(f"Only {len(registered_images)} registered images available, need {num_cameras}")
    
    selected_names = [img.name for _, img in registered_images[:num_cameras]]
    
    # Apply transformation
    transform_params = {
        'scale': scale,
        'rotation': rotation_matrix,
        'translation': np.array(translation)
    }
    
    transformed_reconstruction, _ = apply_transform_to_reconstruction(
        source_reconstruction, transform_params, selected_names
    )
    
    # Save transformed reconstruction
    save_reconstruction(transformed_reconstruction, output_dir)
    
    return {
        'scale': scale,
        'rotation_matrix': rotation_matrix.tolist(),
        'rotation_axis_angle': rotation_axis_angle,
        'translation': translation,
        'num_cameras': num_cameras,
        'selected_cameras': selected_names
    }


def compute_and_apply_similarity_transform(source_dir, target_dir, output_dir, use_robust=False, save_stats=True, quiet=False):
    """
    Compute similarity transform between two COLMAP reconstructions and apply it.
    
    Args:
        source_dir: Path to source COLMAP reconstruction (will be transformed)
        target_dir: Path to target COLMAP reconstruction (reference)
        output_dir: Path to save transformed source reconstruction
        use_robust: Whether to use robust transform (outlier removal)
        save_stats: Whether to save detailed statistics
        quiet: Suppress verbose output
        
    Returns:
        dict: Transform results and statistics
    """
    if not quiet:
        print("=== COLMAP Similarity Transform ===")
        print(f"Source: {source_dir}")
        print(f"Target: {target_dir}")
        print(f"Output: {output_dir}")
        print(f"Robust: {use_robust}")
    
    # Compute similarity transform using existing function
    if not quiet:
        print("\nComputing similarity transform...")
    
    result = compute_similarity_transform(
        source_dir, target_dir, verbose=not quiet, use_robust=use_robust
    )
    
    # Load source reconstruction
    source_reconstruction = load_reconstruction(source_dir)
    if source_reconstruction is None:
        raise ValueError(f"Failed to load source reconstruction from {source_dir}")
    
    # Find common images
    source_poses = extract_camera_poses(source_reconstruction)
    target_poses = extract_camera_poses(load_reconstruction(target_dir))
    common_names = sorted(list(set(source_poses.keys()) & set(target_poses.keys())))
    
    # Apply transform to source reconstruction
    transform_params = {
        'scale': result['scale'],
        'rotation': result['rotation'],
        'translation': result['translation']
    }
    
    transformed_reconstruction, transformed_names = apply_transform_to_reconstruction(
        source_reconstruction, transform_params, common_names
    )
    
    # Save transformed reconstruction
    save_reconstruction(transformed_reconstruction, output_dir)
    
    # Compute post-transform statistics
    if not quiet:
        print("\nComputing post-transform statistics...")
    
    transformed_poses = extract_camera_poses(transformed_reconstruction)
    
    # Calculate final errors
    final_errors = []
    for name in common_names:
        if name in transformed_poses and name in target_poses:
            pos_error = np.linalg.norm(
                transformed_poses[name]['position'] - target_poses[name]['position']
            )
            final_errors.append(pos_error)
    
    final_errors = np.array(final_errors)
    
    # Compile comprehensive results
    stats = {
        'transform_parameters': {
            'scale': float(result['scale']),
            'translation': result['translation'].tolist(),
            'rotation_matrix': result['rotation'].tolist(),
            'rotation_axis_angle': R.from_matrix(result['rotation']).as_rotvec().tolist()
        },
        'alignment_quality': {
            'rmse': float(result['rmse']),
            'all_points_rmse': float(result.get('all_points_rmse', result['rmse'])),
            'final_mean_error': float(np.mean(final_errors)) if len(final_errors) > 0 else None,
            'final_median_error': float(np.median(final_errors)) if len(final_errors) > 0 else None,
            'final_max_error': float(np.max(final_errors)) if len(final_errors) > 0 else None
        },
        'camera_statistics': {
            'num_common_cameras': int(len(common_names)),
            'inlier_count': int(result.get('inlier_count', len(common_names))),
            'outlier_count': int(result.get('outlier_count', 0)),
            'common_cameras': common_names,
            'per_camera_errors': final_errors.tolist() if len(final_errors) > 0 else []
        },
        'processing_info': {
            'source_dir': source_dir,
            'target_dir': target_dir,
            'output_dir': output_dir,
            'used_robust': use_robust,
            'transform_method': 'robust' if use_robust else 'regular'
        }
    }
    
    # Print summary
    if not quiet:
        print(f"\n=== Transform Summary ===")
        print(f"Scale factor: {result['scale']:.6f}")
        print(f"Translation: [{result['translation'][0]:.3f}, {result['translation'][1]:.3f}, {result['translation'][2]:.3f}]")
        print(f"Rotation (axis-angle): [{R.from_matrix(result['rotation']).as_rotvec()[0]:.3f}, {R.from_matrix(result['rotation']).as_rotvec()[1]:.3f}, {R.from_matrix(result['rotation']).as_rotvec()[2]:.3f}]")
        print(f"RMSE: {result['rmse']:.6f}")
        print(f"Common cameras: {len(common_names)}")
        if use_robust:
            print(f"Inliers: {result.get('inlier_count', len(common_names))}")
            print(f"Outliers: {result.get('outlier_count', 0)}")
        print(f"Final mean error: {np.mean(final_errors):.6f}" if len(final_errors) > 0 else "No final error (no common cameras)")
        
        # Quality assessment
        rmse = result['rmse']
        if rmse < 0.1:
            print("✅ Excellent alignment!")
        elif rmse < 0.5:
            print("✅ Good alignment")
        elif rmse < 2.0:
            print("⚠️  Moderate alignment")
        else:
            print("❌ Poor alignment - check data quality")
    
    # Save detailed statistics
    if save_stats:
        stats_file = os.path.join(output_dir, "similarity_transform_stats.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        if not quiet:
            print(f"Detailed statistics saved to: {stats_file}")
    
    return stats


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        if args.test_mode:
            # Test mode: create synthetic data with known transform
            if not args.quiet:
                print("Running in test mode...")
            
            test_info = create_test_data(
                args.source, args.output + "_test_target", 
                args.test_scale, args.test_rotation, args.test_translation, 
                args.test_cameras
            )
            
            # Now compute transform between original and transformed
            stats = compute_and_apply_similarity_transform(
                args.source, args.output + "_test_target", args.output,
                use_robust=args.robust, save_stats=args.save_stats, quiet=args.quiet
            )
            
            # Add test validation info
            stats['test_validation'] = {
                'ground_truth': test_info,
                'is_test_mode': True
            }
            
            # Compare with ground truth
            gt_scale = test_info['scale']
            gt_translation = np.array(test_info['translation'])
            gt_rotation = np.array(test_info['rotation_matrix'])
            
            recovered_scale = stats['transform_parameters']['scale']
            recovered_translation = np.array(stats['transform_parameters']['translation'])
            recovered_rotation = np.array(stats['transform_parameters']['rotation_matrix'])
            
            scale_error = abs(recovered_scale - gt_scale)
            translation_error = np.linalg.norm(recovered_translation - gt_translation)
            rotation_error = np.linalg.norm(recovered_rotation - gt_rotation, 'fro')
            
            stats['test_validation']['errors'] = {
                'scale_error': float(scale_error),
                'translation_error': float(translation_error),
                'rotation_error': float(rotation_error)
            }
            
            if not args.quiet:
                print(f"\n=== Test Validation ===")
                print(f"Scale error: {scale_error:.10f}")
                print(f"Translation error: {translation_error:.10f}")
                print(f"Rotation error: {rotation_error:.10f}")
                
                if scale_error < 1e-10 and translation_error < 1e-10 and rotation_error < 1e-10:
                    print("✅ Perfect recovery - implementation validated!")
                else:
                    print("⚠️  Some numerical errors detected")
            
            # Transform point cloud if provided (in test mode)
            if args.pointcloud:
                if not args.quiet:
                    print(f"\n=== Point Cloud Transformation (Test Mode) ===")
                    print(f"Input point cloud: {args.pointcloud}")
                
                # Use the recovered transform parameters
                transform_params = {
                    'scale': stats['transform_parameters']['scale'],
                    'rotation': np.array(stats['transform_parameters']['rotation_matrix']),
                    'translation': np.array(stats['transform_parameters']['translation'])
                }
                
                # Transform and save point cloud
                success = transform_and_save_pointcloud(
                    args.pointcloud, transform_params, args.output
                )
                
                if success:
                    if not args.quiet:
                        print(f"✅ Point cloud transformation completed successfully!")
                else:
                    if not args.quiet:
                        print(f"❌ Point cloud transformation failed!")
            
            # Generate point cloud from depth maps if provided (in test mode)
            if args.depth_maps and args.confidence_maps:
                if not args.quiet:
                    print(f"\n=== Point Cloud Generation from Depth Maps (Test Mode) ===")
                    print(f"Depth maps directory: {args.depth_maps}")
                    print(f"Confidence maps directory: {args.confidence_maps}")
                
                # Use the recovered transform parameters
                transform_params = {
                    'scale': stats['transform_parameters']['scale'],
                    'rotation': np.array(stats['transform_parameters']['rotation_matrix']),
                    'translation': np.array(stats['transform_parameters']['translation'])
                }
                
                success = generate_pointcloud_from_depth_maps(
                    args.depth_maps, args.confidence_maps, args.source, args.output,
                    conf_threshold=args.conf_threshold, images_dir=args.images,
                    similarity_transform=transform_params
                )
                
                if success:
                    if not args.quiet:
                        print(f"✅ Point cloud generation from depth maps completed successfully!")
                else:
                    if not args.quiet:
                        print(f"❌ Point cloud generation from depth maps failed!")
        
        else:
            # Normal mode: transform between two existing reconstructions
            if not args.target:
                raise ValueError("--target is required when not in test mode")
            
            stats = compute_and_apply_similarity_transform(
                args.source, args.target, args.output,
                use_robust=args.robust, save_stats=args.save_stats, quiet=args.quiet
            )
            
            # Transform point cloud if provided
            if args.pointcloud:
                if not args.quiet:
                    print(f"\n=== Point Cloud Transformation ===")
                    print(f"Input point cloud: {args.pointcloud}")
                
                # Extract transform parameters from stats
                transform_params = {
                    'scale': stats['transform_parameters']['scale'],
                    'rotation': np.array(stats['transform_parameters']['rotation_matrix']),
                    'translation': np.array(stats['transform_parameters']['translation'])
                }
                
                # Transform and save point cloud
                success = transform_and_save_pointcloud(
                    args.pointcloud, transform_params, args.output
                )
                
                if success:
                    if not args.quiet:
                        print(f"✅ Point cloud transformation completed successfully!")
                else:
                    if not args.quiet:
                        print(f"❌ Point cloud transformation failed!")
            
            # Generate point cloud from depth maps if provided
            if args.depth_maps and args.confidence_maps:
                if not args.quiet:
                    print(f"\n=== Point Cloud Generation from Depth Maps ===")
                    print(f"Depth maps directory: {args.depth_maps}")
                    print(f"Confidence maps directory: {args.confidence_maps}")
                
                # Extract transform parameters from stats
                transform_params = {
                    'scale': stats['transform_parameters']['scale'],
                    'rotation': np.array(stats['transform_parameters']['rotation_matrix']),
                    'translation': np.array(stats['transform_parameters']['translation'])
                }
                
                success = generate_pointcloud_from_depth_maps(
                    args.depth_maps, args.confidence_maps, args.source, args.output,
                    conf_threshold=args.conf_threshold, images_dir=args.images, 
                    similarity_transform=transform_params
                )
                
                if success:
                    if not args.quiet:
                        print(f"✅ Point cloud generation from depth maps completed successfully!")
                else:
                    if not args.quiet:
                        print(f"❌ Point cloud generation from depth maps failed!")
        
        if not args.quiet:
            print(f"\n✅ Similarity transform completed successfully!")
            print(f"Transformed reconstruction saved to: {args.output}")
            if args.pointcloud:
                print(f"Transformed point cloud saved to: {os.path.join(args.output, 'aligned.ply')}")
            if args.depth_maps and args.confidence_maps:
                print(f"Generated point cloud saved to: {os.path.join(args.output, 'generated_from_depth.ply')}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 