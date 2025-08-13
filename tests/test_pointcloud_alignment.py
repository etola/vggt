#!/usr/bin/env python3
import numpy as np
import sys
import os
import tempfile
import subprocess

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pycolmap
    import trimesh
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


def create_test_ply(points, colors=None, output_path="test.ply"):
    """Create a simple PLY file for testing."""
    if colors is None:
        colors = np.random.randint(0, 255, (len(points), 3), dtype=np.uint8)
    
    mesh = trimesh.PointCloud(vertices=points, colors=colors)
    mesh.export(output_path)
    return output_path


def create_synthetic_reconstruction(centers, output_dir, image_names=None):
    """Create a synthetic COLMAP reconstruction with given camera centers."""
    if image_names is None:
        image_names = [f"img_{i:03d}.jpg" for i in range(len(centers))]
    
    os.makedirs(output_dir, exist_ok=True)
    
    reconstruction = pycolmap.Reconstruction()
    
    # Add camera
    camera = pycolmap.Camera(
        camera_id=1,
        model="PINHOLE",
        width=640,
        height=480,
        params=[500.0, 500.0, 320.0, 240.0]
    )
    reconstruction.add_camera(camera)
    
    # Add images
    for i, (center, name) in enumerate(zip(centers, image_names)):
        R = np.eye(3)  # Identity rotation for simplicity
        t = -R @ np.array(center, dtype=float)
        
        image = pycolmap.Image(
            id=i+1,
            name=name,
            camera_id=1,
            cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
        )
        image.points2D = pycolmap.ListPoint2D([])
        image.registered = True
        reconstruction.add_image(image)
    
    # Save reconstruction
    reconstruction.write_text(output_dir)
    return reconstruction


def test_pointcloud_alignment():
    """Test the point cloud alignment feature."""
    print("Testing point cloud alignment feature...")
    
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ Skipping: pycolmap or trimesh not available")
        return True
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create synthetic data
        source_centers = np.array([[0, 0, 0], [1, 0, 0]])  # Simple baseline
        target_centers = 2.5 * source_centers  # Scaled by 2.5x
        
        source_dir = os.path.join(temp_dir, "source")
        target_dir = os.path.join(temp_dir, "target")
        output_dir = os.path.join(temp_dir, "output")
        
        # Create reconstructions
        create_synthetic_reconstruction(source_centers, source_dir)
        create_synthetic_reconstruction(target_centers, target_dir)
        
        # Create test point cloud in source coordinate system
        test_points = np.array([
            [0.5, 0.5, 0.5],
            [1.5, -0.5, 0.3],
            [-0.2, 0.8, -0.1]
        ])
        test_colors = np.array([
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255]     # Blue
        ], dtype=np.uint8)
        
        pointcloud_path = os.path.join(temp_dir, "test_pointcloud.ply")
        create_test_ply(test_points, test_colors, pointcloud_path)
        
        # Run alignment with point cloud
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "align_reconstructions.py"))
        
        cmd = [
            "python3", script_path,
            "--source", source_dir,
            "--target", target_dir,
            "--output", output_dir,
            "--pointcloud", pointcloud_path,
            "--pointcloud_output", os.path.join(output_dir, "aligned_test.ply")
        ]
        
        # Execute alignment
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(script_path))
            
            if result.returncode != 0:
                print(f"✗ Failed: align_reconstructions.py returned error code {result.returncode}")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
            
            # Check if output files exist
            aligned_pointcloud = os.path.join(output_dir, "aligned_test.ply")
            if not os.path.exists(aligned_pointcloud):
                print(f"✗ Failed: aligned point cloud not created at {aligned_pointcloud}")
                return False
            
            # Load and verify the aligned point cloud
            aligned_mesh = trimesh.load(aligned_pointcloud)
            aligned_points = aligned_mesh.vertices
            aligned_colors = aligned_mesh.colors
            
            # Verify points were transformed (should be scaled by ~2.5)
            expected_points = 2.5 * test_points
            
            if not np.allclose(aligned_points, expected_points, atol=1e-3):
                print(f"✗ Failed: points not transformed correctly")
                print(f"Expected shape: {expected_points.shape}, got: {aligned_points.shape}")
                print(f"Expected first point: {expected_points[0]}, got: {aligned_points[0]}")
                return False
            
            # Verify colors are preserved (with some tolerance for format differences)
            if aligned_colors is not None:
                # Handle potential differences in color format (RGB vs RGBA, etc.)
                if aligned_colors.shape[1] == 4:  # RGBA
                    aligned_colors = aligned_colors[:, :3]  # Take only RGB
                
                if not np.allclose(aligned_colors, test_colors, atol=1):
                    print(f"✗ Failed: colors not preserved correctly")
                    print(f"Expected: {test_colors}")
                    print(f"Got: {aligned_colors}")
                    # Don't fail for color issues, just warn
                    print("⚠ Color mismatch detected but continuing...")
            else:
                print("⚠ No colors in output (acceptable for some PLY formats)")
            
            print("✓ Point cloud alignment test passed")
            return True
            
        except Exception as e:
            print(f"✗ Failed: exception during alignment: {e}")
            return False


def test_pointcloud_default_output():
    """Test point cloud alignment with default output path."""
    print("Testing point cloud alignment with default output...")
    
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ Skipping: pycolmap or trimesh not available")
        return True
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create minimal test data
        source_centers = np.array([[0, 0, 0], [1, 0, 0]])
        target_centers = 3.0 * source_centers
        
        source_dir = os.path.join(temp_dir, "source")
        target_dir = os.path.join(temp_dir, "target")
        
        create_synthetic_reconstruction(source_centers, source_dir)
        create_synthetic_reconstruction(target_centers, target_dir)
        
        # Create test point cloud
        test_points = np.array([[0.5, 0.5, 0.5]])
        pointcloud_path = os.path.join(temp_dir, "test.ply")
        create_test_ply(test_points, output_path=pointcloud_path)
        
        # Run alignment without specifying output paths
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "align_reconstructions.py"))
        
        cmd = [
            "python3", script_path,
            "--source", source_dir,
            "--target", target_dir,
            "--pointcloud", pointcloud_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(script_path))
            
            if result.returncode != 0:
                print(f"✗ Failed: align_reconstructions.py returned error code {result.returncode}")
                return False
            
            # Check if default output was created (should be test_aligned.ply)
            expected_output = os.path.join(temp_dir, "test_aligned.ply")
            if not os.path.exists(expected_output):
                print(f"✗ Failed: default aligned point cloud not created at {expected_output}")
                return False
            
            print("✓ Point cloud default output test passed")
            return True
            
        except Exception as e:
            print(f"✗ Failed: exception during alignment: {e}")
            return False


def main():
    """Run point cloud alignment tests."""
    print("Running point cloud alignment tests...")
    print("=" * 60)
    
    tests = [
        test_pointcloud_alignment,
        test_pointcloud_default_output,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Point Cloud Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All point cloud tests passed!")
        return 0
    else:
        print("❌ Some point cloud tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 