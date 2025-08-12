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
from scipy.spatial.transform import Rotation as R

# Import functions from utils/similarity_transform.py
from utils.similarity_transform import (
    compute_similarity_transform, 
    load_reconstruction, 
    extract_camera_poses,
    apply_similarity_transform
)


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
        
        else:
            # Normal mode: transform between two existing reconstructions
            if not args.target:
                raise ValueError("--target is required when not in test mode")
            
            stats = compute_and_apply_similarity_transform(
                args.source, args.target, args.output,
                use_robust=args.robust, save_stats=args.save_stats, quiet=args.quiet
            )
        
        if not args.quiet:
            print(f"\n✅ Similarity transform completed successfully!")
            print(f"Transformed reconstruction saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 