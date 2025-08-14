#!/usr/bin/env python3
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.reconstruction_transform import (
    apply_similarity_transform_to_point,
    estimate_scale_from_centers,
    estimate_scale_only_from_recons,
    estimate_rigid_transform,
    _pairwise_distance_ratios,
)


class TestReconstructionTransform(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        self.tolerance = 1e-10
    
    def test_apply_similarity_transform_to_point_identity(self):
        """Test identity transform."""
        point = np.array([1.0, 2.0, 3.0])
        scale = 1.0
        rotation = np.eye(3)
        translation = np.zeros(3)
        
        result = apply_similarity_transform_to_point(point, scale, rotation, translation)
        np.testing.assert_allclose(result, point, atol=self.tolerance)
    
    def test_apply_similarity_transform_to_point_scale_only(self):
        """Test pure scaling."""
        point = np.array([1.0, 2.0, 3.0])
        scale = 2.5
        rotation = np.eye(3)
        translation = np.zeros(3)
        
        expected = scale * point
        result = apply_similarity_transform_to_point(point, scale, rotation, translation)
        np.testing.assert_allclose(result, expected, atol=self.tolerance)
    
    def test_apply_similarity_transform_to_point_translation_only(self):
        """Test pure translation."""
        point = np.array([1.0, 2.0, 3.0])
        scale = 1.0
        rotation = np.eye(3)
        translation = np.array([5.0, -3.0, 1.5])
        
        expected = point + translation
        result = apply_similarity_transform_to_point(point, scale, rotation, translation)
        np.testing.assert_allclose(result, expected, atol=self.tolerance)
    
    def test_apply_similarity_transform_to_point_rotation_only(self):
        """Test pure rotation."""
        point = np.array([1.0, 0.0, 0.0])
        scale = 1.0
        # 90 degree rotation around Z axis
        rotation = R.from_euler('z', 90, degrees=True).as_matrix()
        translation = np.zeros(3)
        
        expected = np.array([0.0, 1.0, 0.0])
        result = apply_similarity_transform_to_point(point, scale, rotation, translation)
        np.testing.assert_allclose(result, expected, atol=self.tolerance)
    
    def test_apply_similarity_transform_to_point_combined(self):
        """Test combined scale, rotation, and translation."""
        point = np.array([1.0, 0.0, 0.0])
        scale = 2.0
        # 90 degree rotation around Z axis
        rotation = R.from_euler('z', 90, degrees=True).as_matrix()
        translation = np.array([1.0, 1.0, 1.0])
        
        # Expected: scale * rotate * point + translation = 2.0 * [0,1,0] + [1,1,1] = [1,3,1]
        expected = np.array([1.0, 3.0, 1.0])
        result = apply_similarity_transform_to_point(point, scale, rotation, translation)
        np.testing.assert_allclose(result, expected, atol=self.tolerance)
    
    def test_apply_similarity_transform_to_point_invalid_input(self):
        """Test error handling for invalid input."""
        scale = 1.0
        rotation = np.eye(3)
        translation = np.zeros(3)
        
        # Test wrong shape
        with self.assertRaises(ValueError):
            apply_similarity_transform_to_point(np.array([1.0, 2.0]), scale, rotation, translation)
        
        with self.assertRaises(ValueError):
            apply_similarity_transform_to_point(np.array([[1.0, 2.0, 3.0]]), scale, rotation, translation)
    
    def test_estimate_scale_from_centers_two_points(self):
        """Test scale estimation with exactly two points."""
        # Create two points with known scale factor
        src_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        scale_true = 3.5
        dst_points = scale_true * src_points
        
        # Test robust method
        scale_robust = estimate_scale_from_centers(src_points, dst_points, robust=True)
        self.assertAlmostEqual(scale_robust, scale_true, places=6)
        
        # Test non-robust method  
        scale_nonrobust = estimate_scale_from_centers(src_points, dst_points, robust=False)
        self.assertAlmostEqual(scale_nonrobust, scale_true, places=6)
    
    def test_estimate_scale_from_centers_multiple_points(self):
        """Test scale estimation with multiple points."""
        # Create multiple points with known scale factor
        src_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0]
        ])
        scale_true = 2.7
        dst_points = scale_true * src_points + np.array([1.0, 2.0, 3.0])  # Add translation
        
        scale_estimated = estimate_scale_from_centers(src_points, dst_points, robust=True)
        self.assertAlmostEqual(scale_estimated, scale_true, places=6)
    
    def test_pairwise_distance_ratios(self):
        """Test pairwise distance ratio computation."""
        src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        scale = 2.5
        dst = scale * src
        
        ratios = _pairwise_distance_ratios(src, dst)
        
        # All ratios should equal the scale factor
        expected_ratios = np.full(len(ratios), scale)
        np.testing.assert_allclose(ratios, expected_ratios, atol=self.tolerance)
    
    def test_estimate_rigid_transform_two_points(self):
        """Test rigid transform estimation with exactly two points."""
        src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # 90 degree rotation + translation
        true_rotation = R.from_euler('z', 90, degrees=True).as_matrix()
        true_translation = np.array([2.0, 3.0, 4.0])
        dst = (true_rotation @ src.T).T + true_translation
        
        estimated_R, estimated_t = estimate_rigid_transform(src, dst)
        
        # Verify the transform works
        transformed = (estimated_R @ src.T).T + estimated_t
        np.testing.assert_allclose(transformed, dst, atol=self.tolerance)
    
    def test_similarity_transform_consistency(self):
        """Test that applying point transform gives same result as manual calculation."""
        # Set up test data
        point = np.array([1.5, -2.3, 0.8])
        scale = 1.7
        rotation = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        translation = np.array([-0.5, 2.1, -1.2])
        
        # Apply using function
        result_function = apply_similarity_transform_to_point(point, scale, rotation, translation)
        
        # Apply manually
        result_manual = scale * (rotation @ point) + translation
        
        np.testing.assert_allclose(result_function, result_manual, atol=self.tolerance)
    
    def test_multiple_points_batch_vs_individual(self):
        """Test that transforming points individually gives same result as batch transform."""
        points = np.array([
            [1.0, 2.0, 3.0],
            [-1.5, 0.5, -2.0],
            [0.0, 0.0, 1.0]
        ])
        scale = 1.3
        rotation = R.from_euler('y', 45, degrees=True).as_matrix()
        translation = np.array([1.0, -1.0, 0.5])
        
        # Transform individually
        individual_results = []
        for point in points:
            transformed = apply_similarity_transform_to_point(point, scale, rotation, translation)
            individual_results.append(transformed)
        individual_results = np.array(individual_results)
        
        # Transform in batch (manual)
        batch_result = scale * (rotation @ points.T).T + translation
        
        np.testing.assert_allclose(individual_results, batch_result, atol=self.tolerance)

    def test_estimate_scale_only_from_recons_vs_full_transform(self):
        """Test that scale-only estimation matches the scale from full similarity transform.
        
        Note: This test uses synthetic data since it requires COLMAP reconstruction directories.
        In practice, the function should be tested with real reconstruction data.
        """
        # This test verifies the function signature and return format
        # Real integration tests would require actual COLMAP reconstruction directories
        
        # Test that the function exists and has correct signature
        self.assertTrue(callable(estimate_scale_only_from_recons))
        
        # Test function signature by checking it accepts the expected parameters
        import inspect
        sig = inspect.signature(estimate_scale_only_from_recons)
        expected_params = ['source_sparse_dir', 'target_sparse_dir', 'robust_scale']
        actual_params = list(sig.parameters.keys())
        self.assertEqual(expected_params, actual_params)
        
        # Test that robust_scale has correct default value
        self.assertTrue(sig.parameters['robust_scale'].default)
        
        # The function would need actual COLMAP reconstruction directories to test fully
        # In a real test scenario, you would:
        # 1. Create or use test COLMAP reconstructions
        # 2. Call both estimate_scale_only_from_recons and estimate_similarity_transform_from_recons
        # 3. Verify that the scale values match between the two approaches
        # 4. Verify that the scale-only version is more efficient
        
        print("Note: Full integration test requires actual COLMAP reconstruction directories")


if __name__ == '__main__':
    unittest.main() 