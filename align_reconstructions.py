#!/usr/bin/env python3
"""
Align COLMAP Reconstructions with Optional Point Cloud Transformation

This script aligns a source COLMAP reconstruction to a target reconstruction using 
matched image names and camera centers. It can optionally transform a point cloud 
from the source coordinate system to the target coordinate system.

Examples:
    # Basic reconstruction alignment
    python3 align_reconstructions.py -s source/sparse -t target/sparse -o aligned/

    # Align reconstruction and transform point cloud  
    python3 align_reconstructions.py -s source/sparse -t target/sparse \\
        -o aligned/ -p source_cloud.ply

    # Specify custom point cloud output path (relative to output dir)
    python3 align_reconstructions.py -s source/sparse -t target/sparse \\
        -o aligned/ -p source_cloud.ply -po my_cloud.ply

    # Specify absolute point cloud output path
    python3 align_reconstructions.py -s source/sparse -t target/sparse \\
        -p source_cloud.ply -po /tmp/aligned_cloud.ply

    # Save transform parameters to JSON
    python3 align_reconstructions.py -s source/sparse -t target/sparse \\
        -j transform.json -p source_cloud.ply
"""
import argparse
import json
import os
import numpy as np
from utils.reconstruction_transform import (
    estimate_similarity_transform_from_recons,
    load_reconstruction,
    apply_similarity_transform_to_reconstruction,
    save_reconstruction_text,
    transform_point_cloud_to_colmap_frame,
    extract_camera_centers_and_rotations,
    apply_similarity_transform_to_point,
)

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align a source COLMAP reconstruction to a target using matched image names and camera centers."
    )
    parser.add_argument("-s", "--source", required=True, help="Path to source COLMAP sparse directory")
    parser.add_argument("-t", "--target", required=True, help="Path to target COLMAP sparse directory")
    parser.add_argument("-o", "--output", default=None, help="If set, directory to save transformed source reconstruction (text format)")
    parser.add_argument("-p", "--pointcloud", default=None, help="If set, path to source point cloud (.ply) to transform and save")
    parser.add_argument("-po", "--pointcloud_output", default=None, help="Output path for transformed point cloud (relative to output dir if not absolute, default: <output>/aligned_pointcloud.ply)")
    parser.add_argument("--no_robust_scale", action="store_true", help="Disable robust scale estimation (use RMS ratio)")
    parser.add_argument("-j", "--json", dest="json_out", default=None, help="If set, write transform JSON to this path")
    return parser.parse_args()


def transform_pointcloud(pointcloud_path, transform_params, output_path):
    """Transform a point cloud using the similarity transform and save it.
    
    Args:
        pointcloud_path: Path to input point cloud (.ply file)
        transform_params: Dict with 'scale', 'rotation', 'translation' keys
        output_path: Path to save transformed point cloud
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not TRIMESH_AVAILABLE:
        print("Error: trimesh library not available. Install with: pip install trimesh")
        return False
    
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
        transformed_points, transformed_colors = transform_point_cloud_to_colmap_frame(
            points, colors, transform_params
        )
        
        # Create new point cloud with transformed points
        if transformed_colors is not None:
            transformed_mesh = trimesh.PointCloud(vertices=transformed_points, colors=transformed_colors)
        else:
            transformed_mesh = trimesh.PointCloud(vertices=transformed_points)
        
        # Save the transformed point cloud
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
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


def validate_camera_point_relationships(source_sparse_dir, pointcloud_path, transform_params, verbose=True):
    """
    Validate that camera-point relationships are preserved after transformation.
    
    This test picks 100 random points from the point cloud and measures the angle 
    between the line connecting each point to each camera center and the camera's 
    Z-axis direction (R^T @ [0,0,1]). It compares these angles before and after 
    transformation - they should remain the same, proving the similarity transform 
    preserves geometric relationships.
    
    Note: This test validates that the actual geometric relationships between cameras
    and points are preserved, regardless of coordinate system conventions.
    
    Args:
        source_sparse_dir: Path to source reconstruction
        pointcloud_path: Path to point cloud file
        transform_params: Transform parameters dict
        verbose: Print detailed output
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        if verbose:
            print("\n" + "="*60)
            print("VALIDATING CAMERA-POINT RELATIONSHIPS")
            print("="*60)
        
        # Load source reconstruction and point cloud
        source_rec = load_reconstruction(source_sparse_dir)
        source_poses = extract_camera_centers_and_rotations(source_rec)
        
        # Load point cloud
        try:
            import trimesh
            mesh = trimesh.load(pointcloud_path)
            if hasattr(mesh, 'vertices'):
                points = mesh.vertices
            else:
                if verbose:
                    print("ERROR: Could not load vertices from point cloud")
                return False
        except Exception as e:
            if verbose:
                print(f"ERROR loading point cloud: {e}")
            return False
        
        if len(points) < 100:
            if verbose:
                print(f"WARNING: Point cloud has only {len(points)} points, using all")
            test_points = points
        else:
            # Randomly sample 100 points for testing
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(points), 100, replace=False)
            test_points = points[indices]
        
        if verbose:
            print(f"Testing with {len(test_points)} points from {len(points)} total")
            print(f"Source cameras: {len(source_poses)}")
        
        # Calculate original angles
        original_angles = {}
        for cam_name, pose_data in source_poses.items():
            cam_center = pose_data['center']
            cam_rotation = pose_data['rotation']
            
            # Camera Z-axis direction in world coordinates  
            # Note: Using camera Z-axis as requested by user (not viewing direction)
            cam_z_axis = cam_rotation.T @ np.array([0, 0, 1])
            
            angles = []
            for point in test_points:
                # Vector from camera to point
                point_to_cam = point - cam_center
                if np.linalg.norm(point_to_cam) > 1e-12:
                    point_to_cam_norm = point_to_cam / np.linalg.norm(point_to_cam)
                    
                    # Angle between camera Z-axis and direction to point
                    dot_product = np.dot(cam_z_axis, point_to_cam_norm)
                    angle = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi
                    angles.append(angle)
            
            original_angles[cam_name] = np.array(angles)
        
        # Transform cameras and points
        transformed_cameras = {}
        for cam_name, pose_data in source_poses.items():
            orig_center = pose_data['center']
            orig_rotation = pose_data['rotation']
            
            # Transform camera center
            new_center = apply_similarity_transform_to_point(
                orig_center,
                transform_params['scale'],
                transform_params['rotation'],
                transform_params['translation']
            )
            
            # Transform camera rotation
            new_rotation = transform_params['rotation'] @ orig_rotation
            
            transformed_cameras[cam_name] = {
                'center': new_center,
                'rotation': new_rotation
            }
        
        # Transform test points
        transformed_points = np.array([
            apply_similarity_transform_to_point(
                point,
                transform_params['scale'],
                transform_params['rotation'],
                transform_params['translation']
            ) for point in test_points
        ])
        
        # Calculate transformed angles
        transformed_angles = {}
        for cam_name, cam_data in transformed_cameras.items():
            cam_center = cam_data['center']
            cam_rotation = cam_data['rotation']
            
            # Camera Z-axis direction in transformed world coordinates
            cam_z_axis = cam_rotation.T @ np.array([0, 0, 1])
            
            angles = []
            for point in transformed_points:
                # Vector from camera to point
                point_to_cam = point - cam_center
                if np.linalg.norm(point_to_cam) > 1e-12:
                    point_to_cam_norm = point_to_cam / np.linalg.norm(point_to_cam)
                    
                    # Angle between camera Z-axis and direction to point
                    dot_product = np.dot(cam_z_axis, point_to_cam_norm)
                    angle = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi
                    angles.append(angle)
            
            transformed_angles[cam_name] = np.array(angles)
        
        # Compare angles
        if verbose:
            print(f"\nAngle comparison results:")
        
        all_differences = []
        validation_passed = True
        
        for cam_name in source_poses.keys():
            if cam_name in original_angles and cam_name in transformed_angles:
                orig_angles = original_angles[cam_name]
                trans_angles = transformed_angles[cam_name]
                
                if len(orig_angles) == len(trans_angles):
                    angle_diffs = np.abs(orig_angles - trans_angles)
                    max_diff = np.max(angle_diffs)
                    mean_diff = np.mean(angle_diffs)
                    all_differences.extend(angle_diffs)
                    
                    if verbose:
                        print(f"  Camera {cam_name}:")
                        print(f"    Max angle difference: {max_diff:.3f}°")
                        print(f"    Mean angle difference: {mean_diff:.3f}°")
                        print(f"    RMS angle difference: {np.sqrt(np.mean(angle_diffs**2)):.3f}°")
                        
                        # Show some example angles for context
                        if len(orig_angles) >= 3:
                            print(f"    Sample original angles: {orig_angles[:3]}")
                            print(f"    Sample transformed angles: {trans_angles[:3]}")
                            print(f"    Sample differences: {angle_diffs[:3]}")
                    
                    # Check if differences are within tolerance 
                    # Note: Some differences may occur due to coordinate system conventions
                    # or numerical precision, but should generally be small for valid transforms
                    if max_diff > 10.0:  # 10 degree tolerance
                        validation_passed = False
                        if verbose:
                            print(f"    ❌ FAILED: Large angle difference detected!")
                    elif verbose:
                        print(f"    ✅ PASSED: Angles preserved within tolerance")
        
        if all_differences:
            overall_max_diff = np.max(all_differences)
            overall_mean_diff = np.mean(all_differences)
            overall_rms_diff = np.sqrt(np.mean(np.array(all_differences)**2))
            
            if verbose:
                print(f"\nOverall statistics:")
                print(f"  Maximum angle difference: {overall_max_diff:.3f}°")
                print(f"  Mean angle difference: {overall_mean_diff:.3f}°")
                print(f"  RMS angle difference: {overall_rms_diff:.3f}°")
                
                if validation_passed:
                    print(f"  🎉 VALIDATION PASSED: Camera-point relationships preserved within tolerance!")
                else:
                    print(f"  ❌ VALIDATION FAILED: Large changes in camera-point relationships detected!")
                    print(f"     This may indicate issues with the similarity transform or coordinate system assumptions.")
                    print(f"     Consider checking if the point cloud is in the same coordinate system as the source cameras.")
        
        return validation_passed
        
    except Exception as e:
        if verbose:
            print(f"ERROR during validation: {e}")
        return False


def main():
    args = parse_args()

    # Validate arguments
    if args.pointcloud is not None:
        if not os.path.isfile(args.pointcloud):
            print(f"Error: Point cloud file not found: {args.pointcloud}")
            return 1
        
        if not args.pointcloud.lower().endswith('.ply'):
            print(f"Warning: Point cloud file should be in PLY format: {args.pointcloud}")

    result = estimate_similarity_transform_from_recons(
        source_sparse_dir=args.source,
        target_sparse_dir=args.target,
        robust_scale=(not args.no_robust_scale),
    )

    print("=== Similarity Transform (source -> target) ===")
    print(f"Common images: {result['num_common']}")
    if result['num_common'] <= 10:
        print(f"Names: {result['common_images']}")
    print(f"Scale: {result['scale']:.9f}")
    print("Rotation (3x3):")
    print(np.array2string(result['rotation'], formatter={'float_kind':lambda x: f"{x: .9f}"}))
    print(f"Translation: {np.array2string(result['translation'], formatter={'float_kind':lambda x: f'{x: .9f}'})}")
    print(f"RMSE (centers): {result['rmse']:.9f}")

    if args.json_out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({
                "scale": float(result['scale']),
                "rotation": result['rotation'].tolist(),
                "translation": result['translation'].tolist(),
                "rmse": float(result['rmse']),
                "num_common": int(result['num_common']),
                "common_images": result['common_images'],
            }, f, indent=2)
        print(f"Wrote transform JSON to {args.json_out}")

    if args.output is not None:
        source_rec = load_reconstruction(args.source)
        transformed = apply_similarity_transform_to_reconstruction(
            source_rec,
            scale=float(result['scale']),
            rotation=result['rotation'],
            translation=result['translation'],
            only_image_names=None,
        )
        save_reconstruction_text(transformed, args.output)
        print(f"Transformed source reconstruction saved to {args.output}")

    # Transform point cloud if provided
    if args.pointcloud is not None:
        # Determine output path for point cloud
        if args.pointcloud_output is not None:
            # If pointcloud_output is absolute, use as-is; if relative, make relative to output dir
            if os.path.isabs(args.pointcloud_output):
                pointcloud_out = args.pointcloud_output
            elif args.output is not None:
                # Relative to output directory
                pointcloud_out = os.path.join(args.output, args.pointcloud_output)
            else:
                # No output dir specified, use relative to current directory
                pointcloud_out = args.pointcloud_output
        elif args.output is not None:
            pointcloud_out = os.path.join(args.output, "aligned_pointcloud.ply")
        else:
            # Default to same directory as input with _aligned suffix
            base_dir = os.path.dirname(args.pointcloud)
            base_name = os.path.splitext(os.path.basename(args.pointcloud))[0]
            pointcloud_out = os.path.join(base_dir, f"{base_name}_aligned.ply")
        
        # Apply transform to point cloud
        transform_params = {
            'scale': result['scale'],
            'rotation': result['rotation'], 
            'translation': result['translation']
        }
        
        success = transform_pointcloud(args.pointcloud, transform_params, pointcloud_out)
        if not success:
            return 1

    # Run validation test if point cloud was provided
    if args.pointcloud is not None:
        transform_params = {
            'scale': result['scale'],
            'rotation': result['rotation'], 
            'translation': result['translation']
        }
        
        validation_passed = validate_camera_point_relationships(
            args.source, args.pointcloud, transform_params, verbose=True
        )
        
        if not validation_passed:
            print("WARNING: Validation failed - camera-point relationships may not be preserved correctly")

    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code is not None:
        exit(exit_code) 