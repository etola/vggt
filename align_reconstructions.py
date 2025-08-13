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

    # Specify custom point cloud output path
    python3 align_reconstructions.py -s source/sparse -t target/sparse \\
        -p source_cloud.ply -po aligned_cloud.ply

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
    parser.add_argument("-po", "--pointcloud_output", default=None, help="Output path for transformed point cloud (default: <output>/aligned_pointcloud.ply)")
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

    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code is not None:
        exit(exit_code) 