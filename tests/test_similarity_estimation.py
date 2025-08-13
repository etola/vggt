#!/usr/bin/env python3
import unittest
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False


class TestSimilarityEstimation(unittest.TestCase):
    """Test similarity transform estimation functions."""
    
    def setUp(self):
        """Set up test data."""
        self.tolerance = 1e-10
        self.loose_tolerance = 1e-6
        
        # Create synthetic point sets for testing
        np.random.seed(42)  # For reproducible tests
        
        # Generate random 3D points
        self.n_points = 8
        self.src_points = np.random.randn(self.n_points, 3) * 2.0
        
        # Known transform parameters
        self.true_scale = 2.5
        self.true_rotation = self._create_rotation_matrix(30, 45, 60)  # Euler angles in degrees
        self.true_translation = np.array([1.5, -2.0, 3.5])
        
        # Apply known transform to create target points
        self.dst_points = self.true_scale * (self.true_rotation @ self.src_points.T).T + self.true_translation
    
    def _create_rotation_matrix(self, rx_deg, ry_deg, rz_deg):
        """Create rotation matrix from Euler angles in degrees."""
        rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
        
        # Rotation around X axis
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        # Rotation around Y axis
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        # Rotation around Z axis
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        return Rz @ Ry @ Rx
    
    def test_pairwise_distance_ratios_perfect_scale(self):
        """Test pairwise distance ratio computation with perfect scaling."""
        from utils.reconstruction_transform import _pairwise_distance_ratios
        
        # Simple case: uniform scaling
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        scale = 3.7
        dst = scale * src
        
        ratios = _pairwise_distance_ratios(src, dst)
        
        # All ratios should equal the scale factor
        expected_ratios = np.full(len(ratios), scale)
        np.testing.assert_allclose(ratios, expected_ratios, atol=self.tolerance)
        
        # Test median equals scale
        self.assertAlmostEqual(np.median(ratios), scale, places=10)
    
    def test_pairwise_distance_ratios_with_noise(self):
        """Test pairwise distance ratios with slight noise (robust estimation)."""
        from utils.reconstruction_transform import _pairwise_distance_ratios
        
        src = self.src_points[:4]  # Use 4 points
        dst = self.true_scale * src + 0.01 * np.random.randn(*src.shape)  # Add small noise
        
        ratios = _pairwise_distance_ratios(src, dst)
        
        # Median should be close to true scale even with noise
        median_ratio = np.median(ratios)
        self.assertAlmostEqual(median_ratio, self.true_scale, places=1)
    
    def test_estimate_scale_two_points_robust(self):
        """Test scale estimation with exactly two points using robust method."""
        from utils.reconstruction_transform import estimate_scale_from_centers
        
        # Use only first two points
        src_two = self.src_points[:2]
        dst_two = self.dst_points[:2]
        
        estimated_scale = estimate_scale_from_centers(src_two, dst_two, robust=True)
        self.assertAlmostEqual(estimated_scale, self.true_scale, places=6)
    
    def test_estimate_scale_two_points_nonrobust(self):
        """Test scale estimation with exactly two points using non-robust method."""
        from utils.reconstruction_transform import estimate_scale_from_centers
        
        # Use only first two points
        src_two = self.src_points[:2]
        dst_two = self.dst_points[:2]
        
        estimated_scale = estimate_scale_from_centers(src_two, dst_two, robust=False)
        self.assertAlmostEqual(estimated_scale, self.true_scale, places=6)
    
    def test_estimate_scale_multiple_points(self):
        """Test scale estimation with multiple points."""
        from utils.reconstruction_transform import estimate_scale_from_centers
        
        # Test both robust and non-robust methods
        for robust in [True, False]:
            with self.subTest(robust=robust):
                estimated_scale = estimate_scale_from_centers(self.src_points, self.dst_points, robust=robust)
                self.assertAlmostEqual(estimated_scale, self.true_scale, places=6)
    
    def test_estimate_scale_degenerate_cases(self):
        """Test scale estimation error handling for degenerate cases."""
        from utils.reconstruction_transform import estimate_scale_from_centers
        
        # Single point should fail
        with self.assertRaises(ValueError):
            estimate_scale_from_centers(np.array([[1, 2, 3]]), np.array([[2, 4, 6]]))
        
        # Points too close together (non-robust case)
        close_points = np.array([[0, 0, 0], [1e-15, 0, 0]])
        scaled_close = 2.0 * close_points
        
        with self.assertRaises(ValueError):
            estimate_scale_from_centers(close_points, scaled_close, robust=False)
    
    def test_estimate_rigid_transform_two_points(self):
        """Test rigid transform estimation with exactly two points."""
        from utils.reconstruction_transform import estimate_rigid_transform
        
        # Use scaled source points (scale already applied)
        scaled_src = self.true_scale * self.src_points
        
        # Use only first two points
        src_two = scaled_src[:2]
        dst_two = self.dst_points[:2]
        
        estimated_R, estimated_t = estimate_rigid_transform(src_two, dst_two)
        
        # Verify the transform works
        transformed = (estimated_R @ src_two.T).T + estimated_t
        np.testing.assert_allclose(transformed, dst_two, atol=self.loose_tolerance)
    
    def test_estimate_rigid_transform_multiple_points(self):
        """Test rigid transform estimation with multiple points."""
        from utils.reconstruction_transform import estimate_rigid_transform
        
        # Use scaled source points (scale already applied)
        scaled_src = self.true_scale * self.src_points
        
        estimated_R, estimated_t = estimate_rigid_transform(scaled_src, self.dst_points)
        
        # Verify the transform works
        transformed = (estimated_R @ scaled_src.T).T + estimated_t
        np.testing.assert_allclose(transformed, self.dst_points, atol=self.loose_tolerance)
        
        # Check if estimated parameters are close to true values
        np.testing.assert_allclose(estimated_R, self.true_rotation, atol=self.loose_tolerance)
        np.testing.assert_allclose(estimated_t, self.true_translation, atol=self.loose_tolerance)
    
    def test_estimate_rigid_transform_orthogonality(self):
        """Test that estimated rotation matrix is orthogonal."""
        from utils.reconstruction_transform import estimate_rigid_transform
        
        scaled_src = self.true_scale * self.src_points
        estimated_R, estimated_t = estimate_rigid_transform(scaled_src, self.dst_points)
        
        # Check orthogonality: R @ R.T = I
        should_be_identity = estimated_R @ estimated_R.T
        np.testing.assert_allclose(should_be_identity, np.eye(3), atol=self.loose_tolerance)
        
        # Check determinant is +1 (proper rotation)
        det = np.linalg.det(estimated_R)
        self.assertAlmostEqual(det, 1.0, places=6)
    
    def test_estimate_rigid_transform_degenerate_case(self):
        """Test rigid transform estimation error handling."""
        from utils.reconstruction_transform import estimate_rigid_transform
        
        # Single point should fail
        with self.assertRaises(ValueError):
            estimate_rigid_transform(np.array([[1, 2, 3]]), np.array([[2, 4, 6]]))
    
    def test_synthetic_similarity_transform_estimation(self):
        """Test end-to-end similarity transform estimation with synthetic data."""
        from utils.reconstruction_transform import (
            estimate_scale_from_centers,
            estimate_rigid_transform,
            apply_similarity_transform_to_point
        )
        
        # Step 1: Estimate scale
        estimated_scale = estimate_scale_from_centers(self.src_points, self.dst_points, robust=True)
        
        # Step 2: Scale source points and estimate rigid transform
        scaled_src = estimated_scale * self.src_points
        estimated_R, estimated_t = estimate_rigid_transform(scaled_src, self.dst_points)
        
        # Step 3: Verify parameters are close to ground truth
        self.assertAlmostEqual(estimated_scale, self.true_scale, places=6)
        np.testing.assert_allclose(estimated_R, self.true_rotation, atol=self.loose_tolerance)
        np.testing.assert_allclose(estimated_t, self.true_translation, atol=self.loose_tolerance)
        
        # Step 4: Verify transform works by applying to all points
        for i, src_point in enumerate(self.src_points):
            transformed = apply_similarity_transform_to_point(
                src_point, estimated_scale, estimated_R, estimated_t
            )
            np.testing.assert_allclose(transformed, self.dst_points[i], atol=self.loose_tolerance)
    
    def test_minimal_case_two_points(self):
        """Test similarity transform estimation with minimal case (2 points)."""
        from utils.reconstruction_transform import (
            estimate_scale_from_centers,
            estimate_rigid_transform
        )
        
        # Use only two points
        src_two = self.src_points[:2]
        dst_two = self.dst_points[:2]
        
        # Estimate transform
        estimated_scale = estimate_scale_from_centers(src_two, dst_two, robust=True)
        scaled_src = estimated_scale * src_two
        estimated_R, estimated_t = estimate_rigid_transform(scaled_src, dst_two)
        
        # Verify it perfectly aligns the two points
        transformed = estimated_scale * (estimated_R @ src_two.T).T + estimated_t
        np.testing.assert_allclose(transformed, dst_two, atol=self.tolerance)
    
    def test_identity_transform_estimation(self):
        """Test estimation when true transform is identity."""
        from utils.reconstruction_transform import (
            estimate_scale_from_centers,
            estimate_rigid_transform
        )
        
        # Identity transform case
        src = self.src_points
        dst = self.src_points.copy()  # Same points
        
        estimated_scale = estimate_scale_from_centers(src, dst, robust=True)
        scaled_src = estimated_scale * src
        estimated_R, estimated_t = estimate_rigid_transform(scaled_src, dst)
        
        # Should recover identity transform
        self.assertAlmostEqual(estimated_scale, 1.0, places=6)
        np.testing.assert_allclose(estimated_R, np.eye(3), atol=self.loose_tolerance)
        np.testing.assert_allclose(estimated_t, np.zeros(3), atol=self.loose_tolerance)
    
    def test_pure_scale_estimation(self):
        """Test estimation when transform is pure scaling."""
        from utils.reconstruction_transform import (
            estimate_scale_from_centers,
            estimate_rigid_transform
        )
        
        # Pure scaling case
        scale_only = 4.2
        src = self.src_points
        dst = scale_only * src
        
        estimated_scale = estimate_scale_from_centers(src, dst, robust=True)
        scaled_src = estimated_scale * src
        estimated_R, estimated_t = estimate_rigid_transform(scaled_src, dst)
        
        # Should recover pure scale transform
        self.assertAlmostEqual(estimated_scale, scale_only, places=6)
        np.testing.assert_allclose(estimated_R, np.eye(3), atol=self.loose_tolerance)
        np.testing.assert_allclose(estimated_t, np.zeros(3), atol=self.loose_tolerance)
    
    def test_pure_translation_estimation(self):
        """Test estimation when transform is pure translation."""
        from utils.reconstruction_transform import (
            estimate_scale_from_centers,
            estimate_rigid_transform
        )
        
        # Pure translation case
        translation_only = np.array([5.0, -3.0, 2.0])
        src = self.src_points
        dst = src + translation_only
        
        estimated_scale = estimate_scale_from_centers(src, dst, robust=True)
        scaled_src = estimated_scale * src
        estimated_R, estimated_t = estimate_rigid_transform(scaled_src, dst)
        
        # Should recover pure translation transform
        self.assertAlmostEqual(estimated_scale, 1.0, places=6)
        np.testing.assert_allclose(estimated_R, np.eye(3), atol=self.loose_tolerance)
        np.testing.assert_allclose(estimated_t, translation_only, atol=self.loose_tolerance)

    @unittest.skipUnless(PYCOLMAP_AVAILABLE, "pycolmap not available")
    def test_synthetic_reconstruction_transform(self):
        """Test similarity transform estimation with synthetic COLMAP reconstructions."""
        from utils.reconstruction_transform import (
            estimate_similarity_transform_from_recons,
            save_reconstruction_text
        )
        
        # Create synthetic camera poses
        camera_centers = np.array([
            [0, 0, 0],
            [1, 0, 0], 
            [0, 1, 0],
            [1, 1, 0]
        ], dtype=float)
        
        # Apply known transform
        transformed_centers = self.true_scale * (self.true_rotation @ camera_centers.T).T + self.true_translation
        
        # Create temporary directories for synthetic reconstructions
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, "source")
            target_dir = os.path.join(temp_dir, "target")
            os.makedirs(source_dir)
            os.makedirs(target_dir)
            
            # Create synthetic source reconstruction
            source_rec = pycolmap.Reconstruction()
            
            # Add a camera
            camera = pycolmap.Camera(
                camera_id=1,
                model="PINHOLE",
                width=640,
                height=480,
                params=[500.0, 500.0, 320.0, 240.0]
            )
            source_rec.add_camera(camera)
            
            # Add images with poses
            for i, center in enumerate(camera_centers):
                # Create pose from center (assume looking down -Z)
                R = np.eye(3)  # Identity rotation for simplicity
                t = -R @ center  # Convert center to translation
                
                image = pycolmap.Image(
                    id=i+1,
                    name=f"image_{i:03d}.jpg",
                    camera_id=1,
                    cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
                )
                image.points2D = pycolmap.ListPoint2D([])
                image.registered = True
                source_rec.add_image(image)
            
            # Create synthetic target reconstruction  
            target_rec = pycolmap.Reconstruction()
            target_rec.add_camera(camera)
            
            for i, center in enumerate(transformed_centers):
                R = self.true_rotation  # Apply rotation to camera orientation
                t = -R @ center
                
                image = pycolmap.Image(
                    id=i+1,
                    name=f"image_{i:03d}.jpg",
                    camera_id=1,
                    cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
                )
                image.points2D = pycolmap.ListPoint2D([])
                image.registered = True
                target_rec.add_image(image)
            
            # Save reconstructions
            save_reconstruction_text(source_rec, source_dir)
            save_reconstruction_text(target_rec, target_dir)
            
            # Test the estimation function
            result = estimate_similarity_transform_from_recons(
                source_dir, target_dir, robust_scale=True
            )
            
            # Verify estimated parameters
            self.assertAlmostEqual(result['scale'], self.true_scale, places=3)
            self.assertEqual(result['num_common'], 4)
            self.assertLess(result['rmse'], 1e-6)
            self.assertLess(result['validation_rmse'], 1e-6)
            
            # Check that all image names are found
            expected_names = [f"image_{i:03d}.jpg" for i in range(4)]
            self.assertEqual(sorted(result['common_images']), sorted(expected_names))


def main():
    """Run all similarity estimation tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    main() 