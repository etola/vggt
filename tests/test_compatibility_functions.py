#!/usr/bin/env python3
import numpy as np
import sys
import os
import tempfile

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False


class TestCompatibilityFunctions:
    """Test compatibility functions added to reconstruction_transform.py"""
    
    def __init__(self):
        self.tolerance = 1e-10
        self.loose_tolerance = 1e-6
        
        # Create test data
        np.random.seed(42)
        self.points = np.random.randn(5, 3) * 2.0
        self.colors = np.random.randint(0, 255, (5, 3), dtype=np.uint8)
        
        # Test transform parameters
        self.transform = {
            'scale': 2.5,
            'rotation': self._create_rotation_matrix(30, 45, 60),
            'translation': np.array([1.5, -2.0, 3.5])
        }
    
    def _create_rotation_matrix(self, rx_deg, ry_deg, rz_deg):
        """Create rotation matrix from Euler angles in degrees."""
        rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
        
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        return Rz @ Ry @ Rx
    
    def test_apply_similarity_transform_batch(self):
        """Test apply_similarity_transform function with multiple points."""
        print("Testing apply_similarity_transform (batch)...")
        
        from utils.reconstruction_transform import apply_similarity_transform
        
        # Test with multiple points
        result = apply_similarity_transform(self.points, self.transform)
        
        # Verify shape
        if result.shape != self.points.shape:
            print(f"✗ Failed: shape mismatch {result.shape} != {self.points.shape}")
            return False
        
        # Verify transform is applied correctly
        expected = self.transform['scale'] * (self.transform['rotation'] @ self.points.T).T + self.transform['translation']
        if not np.allclose(result, expected, atol=self.tolerance):
            print(f"✗ Failed: transform not applied correctly")
            return False
        
        print("✓ apply_similarity_transform (batch) test passed")
        return True
    
    def test_apply_similarity_transform_empty(self):
        """Test apply_similarity_transform with empty array."""
        print("Testing apply_similarity_transform (empty array)...")
        
        from utils.reconstruction_transform import apply_similarity_transform
        
        empty_points = np.zeros((0, 3))
        result = apply_similarity_transform(empty_points, self.transform)
        
        if result.shape != (0, 3):
            print(f"✗ Failed: empty array should return empty array, got {result.shape}")
            return False
        
        print("✓ apply_similarity_transform (empty array) test passed")
        return True
    
    def test_apply_similarity_transform_single_point(self):
        """Test apply_similarity_transform with single point."""
        print("Testing apply_similarity_transform (single point)...")
        
        from utils.reconstruction_transform import (
            apply_similarity_transform, 
            apply_similarity_transform_to_point
        )
        
        single_point = self.points[:1]  # Shape (1, 3)
        
        # Test batch function
        result_batch = apply_similarity_transform(single_point, self.transform)
        
        # Test single point function for comparison
        result_single = apply_similarity_transform_to_point(
            single_point[0], 
            self.transform['scale'], 
            self.transform['rotation'], 
            self.transform['translation']
        )
        
        # Should give same result
        if not np.allclose(result_batch[0], result_single, atol=self.tolerance):
            print(f"✗ Failed: batch and single point functions disagree")
            return False
        
        print("✓ apply_similarity_transform (single point) test passed")
        return True
    
    def test_transform_point_cloud_to_colmap_frame(self):
        """Test transform_point_cloud_to_colmap_frame function."""
        print("Testing transform_point_cloud_to_colmap_frame...")
        
        from utils.reconstruction_transform import transform_point_cloud_to_colmap_frame
        
        transformed_points, transformed_colors = transform_point_cloud_to_colmap_frame(
            self.points, self.colors, self.transform
        )
        
        # Check that colors are unchanged
        if not np.array_equal(transformed_colors, self.colors):
            print(f"✗ Failed: colors should be unchanged")
            return False
        
        # Check that points are transformed correctly
        expected_points = self.transform['scale'] * (self.transform['rotation'] @ self.points.T).T + self.transform['translation']
        if not np.allclose(transformed_points, expected_points, atol=self.tolerance):
            print(f"✗ Failed: points not transformed correctly")
            return False
        
        # Check return types
        if not isinstance(transformed_points, np.ndarray) or not isinstance(transformed_colors, np.ndarray):
            print(f"✗ Failed: should return numpy arrays")
            return False
        
        print("✓ transform_point_cloud_to_colmap_frame test passed")
        return True
    
    def test_transform_point_cloud_empty(self):
        """Test transform_point_cloud_to_colmap_frame with empty arrays."""
        print("Testing transform_point_cloud_to_colmap_frame (empty)...")
        
        from utils.reconstruction_transform import transform_point_cloud_to_colmap_frame
        
        empty_points = np.zeros((0, 3))
        empty_colors = np.zeros((0, 3), dtype=np.uint8)
        
        transformed_points, transformed_colors = transform_point_cloud_to_colmap_frame(
            empty_points, empty_colors, self.transform
        )
        
        if transformed_points.shape != (0, 3) or transformed_colors.shape != (0, 3):
            print(f"✗ Failed: empty arrays should return empty arrays")
            return False
        
        print("✓ transform_point_cloud_to_colmap_frame (empty) test passed")
        return True
    
    def test_compute_similarity_transform_wrapper(self):
        """Test compute_similarity_transform compatibility wrapper."""
        print("Testing compute_similarity_transform wrapper...")
        
        if not PYCOLMAP_AVAILABLE:
            print("⚠ Skipping: pycolmap not available")
            return True
        
        from utils.reconstruction_transform import (
            compute_similarity_transform,
            estimate_similarity_transform_from_recons,
            save_reconstruction_text
        )
        
        # Create synthetic test data
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, "source")
            target_dir = os.path.join(temp_dir, "target")
            os.makedirs(source_dir)
            os.makedirs(target_dir)
            
            # Create minimal reconstructions
            source_rec = pycolmap.Reconstruction()
            target_rec = pycolmap.Reconstruction()
            
            # Add camera
            camera = pycolmap.Camera(
                camera_id=1,
                model="PINHOLE",
                width=640,
                height=480,
                params=[500.0, 500.0, 320.0, 240.0]
            )
            source_rec.add_camera(camera)
            target_rec.add_camera(camera)
            
            # Add images with different poses
            for i, center in enumerate([[0, 0, 0], [1, 0, 0]]):
                R = np.eye(3)
                t = -R @ np.array(center, dtype=float)
                
                source_image = pycolmap.Image(
                    id=i+1,
                    name=f"img_{i:03d}.jpg",
                    camera_id=1,
                    cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
                )
                source_image.points2D = pycolmap.ListPoint2D([])
                source_image.registered = True
                source_rec.add_image(source_image)
                
                # Target has scaled positions
                scaled_center = 2.0 * np.array(center, dtype=float)
                t_target = -R @ scaled_center
                
                target_image = pycolmap.Image(
                    id=i+1,
                    name=f"img_{i:03d}.jpg",
                    camera_id=1,
                    cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(R), t_target)
                )
                target_image.points2D = pycolmap.ListPoint2D([])
                target_image.registered = True
                target_rec.add_image(target_image)
            
            # Save reconstructions
            save_reconstruction_text(source_rec, source_dir)
            save_reconstruction_text(target_rec, target_dir)
            
            # Test wrapper function
            result_wrapper = compute_similarity_transform(source_dir, target_dir, verbose=False, use_robust=True)
            
            # Test direct function
            result_direct = estimate_similarity_transform_from_recons(source_dir, target_dir, robust_scale=True)
            
            # Should give same results
            if abs(result_wrapper['scale'] - result_direct['scale']) > self.loose_tolerance:
                print(f"✗ Failed: wrapper and direct function give different scales")
                return False
            
            if not np.allclose(result_wrapper['rotation'], result_direct['rotation'], atol=self.loose_tolerance):
                print(f"✗ Failed: wrapper and direct function give different rotations")
                return False
            
            # Check expected scale (should be ~2.0)
            if abs(result_wrapper['scale'] - 2.0) > 0.1:
                print(f"✗ Failed: expected scale ~2.0, got {result_wrapper['scale']}")
                return False
        
        print("✓ compute_similarity_transform wrapper test passed")
        return True
    
    def test_extract_camera_poses_wrapper(self):
        """Test extract_camera_poses compatibility wrapper."""
        print("Testing extract_camera_poses wrapper...")
        
        if not PYCOLMAP_AVAILABLE:
            print("⚠ Skipping: pycolmap not available")
            return True
        
        from utils.reconstruction_transform import (
            extract_camera_poses,
            extract_camera_centers_and_rotations
        )
        
        # Create test reconstruction
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
        test_centers = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        for i, center in enumerate(test_centers):
            R = np.eye(3)
            t = -R @ np.array(center, dtype=float)
            
            image = pycolmap.Image(
                id=i+1,
                name=f"test_{i:03d}.jpg",
                camera_id=1,
                cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
            )
            image.points2D = pycolmap.ListPoint2D([])
            image.registered = True
            reconstruction.add_image(image)
        
        # Test wrapper function
        poses_wrapper = extract_camera_poses(reconstruction)
        
        # Test direct function
        poses_direct = extract_camera_centers_and_rotations(reconstruction)
        
        # Check that wrapper converts 'center' to 'position'
        for name in poses_wrapper.keys():
            if 'position' not in poses_wrapper[name]:
                print(f"✗ Failed: wrapper should have 'position' key")
                return False
            
            if 'center' in poses_wrapper[name]:
                print(f"✗ Failed: wrapper should not have 'center' key")
                return False
            
            # Check that values match
            if not np.allclose(poses_wrapper[name]['position'], poses_direct[name]['center'], atol=self.tolerance):
                print(f"✗ Failed: position values don't match")
                return False
        
        # Check number of poses
        if len(poses_wrapper) != len(test_centers):
            print(f"✗ Failed: expected {len(test_centers)} poses, got {len(poses_wrapper)}")
            return False
        
        print("✓ extract_camera_poses wrapper test passed")
        return True
    
    def test_function_signatures(self):
        """Test that all new functions have correct signatures."""
        print("Testing function signatures...")
        
        from utils.reconstruction_transform import (
            apply_similarity_transform,
            transform_point_cloud_to_colmap_frame,
            compute_similarity_transform,
            extract_camera_poses
        )
        
        # Check that functions exist and are callable
        functions = [
            apply_similarity_transform,
            transform_point_cloud_to_colmap_frame, 
            compute_similarity_transform,
            extract_camera_poses
        ]
        
        for func in functions:
            if not callable(func):
                print(f"✗ Failed: {func.__name__} is not callable")
                return False
        
        print("✓ Function signatures test passed")
        return True


def main():
    """Run all compatibility function tests."""
    print("Running compatibility function tests...")
    print("=" * 70)
    
    tester = TestCompatibilityFunctions()
    
    tests = [
        tester.test_apply_similarity_transform_batch,
        tester.test_apply_similarity_transform_empty,
        tester.test_apply_similarity_transform_single_point,
        tester.test_transform_point_cloud_to_colmap_frame,
        tester.test_transform_point_cloud_empty,
        tester.test_compute_similarity_transform_wrapper,
        tester.test_extract_camera_poses_wrapper,
        tester.test_function_signatures,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 70)
    print(f"Compatibility Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All compatibility tests passed!")
        return 0
    else:
        print("❌ Some compatibility tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 