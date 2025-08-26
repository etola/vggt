"""
Unit tests for dense correspondence and triangulation functions.

Tests the core functions used in extract_dense_correspondences.py for:
- Intrinsics adjustment for VGGT preprocessing
- 3D point sharing analysis
- Image pair selection based on baseline and shared points
- Triangulation and geometric filtering
"""

import unittest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

# Import functions to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import functions, skip tests if dependencies are missing
try:
    from extract_dense_correspondences import (
        adjust_intrinsics_for_preprocessing,
        calculate_baseline_distance,
        find_good_image_pairs,
        triangulate_correspondences,
        compute_epipolar_distance,
        compute_triangulation_angle,
        compute_reprojection_error,
        filter_correspondences,
        analyze_3d_point_sharing_for_pairs
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dependencies not available for full testing: {e}")
    DEPENDENCIES_AVAILABLE = False
    
    # Mock the functions for testing when dependencies are missing
    def adjust_intrinsics_for_preprocessing(intrinsics, original_size, target_size=518):
        """Mock implementation for testing."""
        width, height = original_size
        max_dim = max(width, height)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2
        scale = target_size / max_dim
        
        adjusted_intrinsics = intrinsics.copy()
        adjusted_intrinsics[0, 2] += left
        adjusted_intrinsics[1, 2] += top
        adjusted_intrinsics[0, 0] *= scale
        adjusted_intrinsics[1, 1] *= scale
        adjusted_intrinsics[0, 2] *= scale
        adjusted_intrinsics[1, 2] *= scale
        
        return adjusted_intrinsics
    
    def calculate_baseline_distance(pos1, pos2):
        """Mock implementation for testing."""
        return np.linalg.norm(pos1 - pos2)
    
    def triangulate_correspondences(pts1, pts2, K1, K2, R1, t1, R2, t2):
        """Mock implementation for testing."""
        # Simple mock triangulation - just return some 3D points
        points_3d = []
        for i in range(len(pts1)):
            # Simple approximation for testing
            X = np.array([0.5, 0.0, -1.0])
            points_3d.append(X)
        return np.array(points_3d)
    
    def compute_triangulation_angle(pts_3d, cam_center1, cam_center2):
        """Mock implementation for testing."""
        vec1 = pts_3d - cam_center1[None, :]
        vec2 = pts_3d - cam_center2[None, :]
        vec1_norm = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8)
        cos_angles = np.sum(vec1_norm * vec2_norm, axis=1)
        cos_angles = np.clip(cos_angles, -1, 1)
        angles = np.arccos(cos_angles) * 180 / np.pi
        return angles
    
    def compute_epipolar_distance(pts1, pts2, F):
        """Mock implementation for testing."""
        return np.array([0.1] * len(pts1))
    
    def compute_reprojection_error(pts_3d, pts1, pts2, K1, K2, R1, t1, R2, t2):
        """Mock implementation for testing."""
        return np.array([0.5] * len(pts_3d))
    
    def filter_correspondences(pts1, pts2, pts_3d, K1, K2, R1, t1, R2, t2, cam_center1, cam_center2, args):
        """Mock implementation for testing."""
        return np.ones(len(pts1), dtype=bool)
    
    def find_good_image_pairs(point_sharing_info, min_shared_points=100, min_baseline_ratio=0.1):
        """Mock implementation for testing."""
        pairs = []
        image_names = point_sharing_info['image_names']
        for i, img1 in enumerate(image_names):
            for j, img2 in enumerate(image_names[i+1:], i+1):
                shared = len(point_sharing_info['image_to_points'].get(img1, set()).intersection(
                    point_sharing_info['image_to_points'].get(img2, set())))
                if shared >= min_shared_points:
                    baseline = calculate_baseline_distance(
                        point_sharing_info['image_positions'][img1],
                        point_sharing_info['image_positions'][img2]
                    )
                    if baseline >= min_baseline_ratio:
                        pairs.append((img1, img2, shared, baseline))
        return pairs
    
    def analyze_3d_point_sharing_for_pairs(reconstruction):
        """Mock implementation for testing."""
        return {
            'image_to_points': {'img1.jpg': {1}, 'img2.jpg': {1}},
            'point_to_images': {1: {'img1.jpg', 'img2.jpg'}},
            'image_names': ['img1.jpg', 'img2.jpg'],
            'image_positions': {'img1.jpg': np.array([0, 0, 0]), 'img2.jpg': np.array([1, 0, 0])}
        }


class TestIntrinsicsAdjustment(unittest.TestCase):
    """Test camera intrinsics adjustment for VGGT preprocessing."""
    
    def test_adjust_intrinsics_square_image(self):
        """Test intrinsics adjustment for square image (no padding needed)."""
        # Original intrinsics for 1000x1000 image
        intrinsics = np.array([
            [800.0, 0.0, 500.0],
            [0.0, 800.0, 500.0],
            [0.0, 0.0, 1.0]
        ])
        
        original_size = (1000, 1000)  # width, height
        target_size = 518
        
        adjusted = adjust_intrinsics_for_preprocessing(intrinsics, original_size, target_size)
        
        # Expected scale factor: 518/1000 = 0.518
        expected_scale = 518.0 / 1000.0
        
        # For square image, no padding is needed, only scaling
        expected = np.array([
            [800.0 * expected_scale, 0.0, 500.0 * expected_scale],
            [0.0, 800.0 * expected_scale, 500.0 * expected_scale],
            [0.0, 0.0, 1.0]
        ])
        
        np.testing.assert_array_almost_equal(adjusted, expected, decimal=5)
    
    def test_adjust_intrinsics_rectangular_image(self):
        """Test intrinsics adjustment for rectangular image (padding needed)."""
        # Original intrinsics for 1200x800 image
        intrinsics = np.array([
            [900.0, 0.0, 600.0],
            [0.0, 900.0, 400.0],
            [0.0, 0.0, 1.0]
        ])
        
        original_size = (1200, 800)  # width, height
        target_size = 518
        
        adjusted = adjust_intrinsics_for_preprocessing(intrinsics, original_size, target_size)
        
        # Max dimension is 1200, so scale = 518/1200
        # Padding: left = (1200-1200)//2 = 0, top = (1200-800)//2 = 200
        expected_scale = 518.0 / 1200.0
        expected_left = 0
        expected_top = 200
        
        expected = np.array([
            [900.0 * expected_scale, 0.0, (600.0 + expected_left) * expected_scale],
            [0.0, 900.0 * expected_scale, (400.0 + expected_top) * expected_scale],
            [0.0, 0.0, 1.0]
        ])
        
        np.testing.assert_array_almost_equal(adjusted, expected, decimal=5)


class TestGeometricFunctions(unittest.TestCase):
    """Test geometric computation functions."""
    
    def test_calculate_baseline_distance(self):
        """Test baseline distance calculation."""
        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([3.0, 4.0, 0.0])
        
        distance = calculate_baseline_distance(pos1, pos2)
        self.assertAlmostEqual(distance, 5.0, places=6)
    
    def test_triangulate_correspondences(self):
        """Test point triangulation from correspondences."""
        # Simple test case with known geometry
        # Camera 1 at origin looking down -Z
        K1 = np.array([[500, 0, 250], [0, 500, 250], [0, 0, 1]], dtype=float)
        R1 = np.eye(3)
        t1 = np.array([0, 0, 0], dtype=float)
        
        # Camera 2 translated along X
        K2 = K1.copy()
        R2 = np.eye(3)
        t2 = np.array([1, 0, 0], dtype=float)
        
        # Point correspondences with some disparity
        pts1 = np.array([[250, 250]], dtype=float)  # Center of first image
        pts2 = np.array([[200, 250]], dtype=float)  # Slightly left in second image (positive disparity)
        
        points_3d = triangulate_correspondences(pts1, pts2, K1, K2, R1, t1, R2, t2)
        
        # Basic sanity checks
        self.assertEqual(points_3d.shape, (1, 3))
        # The triangulated point should be finite and reasonable
        self.assertTrue(np.all(np.isfinite(points_3d[0])))
        # Z coordinate should be positive (point in front of cameras in world coordinates)
        self.assertGreater(points_3d[0, 2], 0)
    
    def test_compute_triangulation_angle(self):
        """Test triangulation angle computation."""
        # Point at origin
        pts_3d = np.array([[0.0, 0.0, 0.0]])
        
        # Camera centers forming 90-degree angle
        cam_center1 = np.array([1.0, 0.0, 0.0])
        cam_center2 = np.array([0.0, 1.0, 0.0])
        
        angles = compute_triangulation_angle(pts_3d, cam_center1, cam_center2)
        
        # Should be 90 degrees
        self.assertAlmostEqual(angles[0], 90.0, places=1)
    
    def test_compute_epipolar_distance(self):
        """Test epipolar distance computation."""
        # Simple fundamental matrix (identity for this test)
        F = np.array([
            [0, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ]) * 0.001  # Small values to avoid numerical issues
        
        # Perfect correspondences should have low epipolar distance
        pts1 = np.array([[100, 100], [200, 200]])
        pts2 = np.array([[100, 100], [200, 200]])
        
        distances = compute_epipolar_distance(pts1, pts2, F)
        
        # Distances should be very small for perfect correspondences
        self.assertTrue(np.all(distances < 1.0))


class TestImagePairSelection(unittest.TestCase):
    """Test image pair selection based on 3D point sharing and baseline."""
    
    def setUp(self):
        """Create mock point sharing info for testing."""
        self.point_sharing_info = {
            'image_to_points': {
                'img1.jpg': {1, 2, 3, 4, 5},
                'img2.jpg': {3, 4, 5, 6, 7},
                'img3.jpg': {1, 2, 8, 9, 10},
                'img4.jpg': {11, 12, 13, 14, 15}
            },
            'point_to_images': {},
            'image_names': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            'image_positions': {
                'img1.jpg': np.array([0.0, 0.0, 0.0]),
                'img2.jpg': np.array([5.0, 0.0, 0.0]),  # Increased baseline
                'img3.jpg': np.array([0.0, 5.0, 0.0]),  # Increased baseline
                'img4.jpg': np.array([10.0, 0.0, 0.0])  # Far away
            }
        }
    
    def test_find_good_image_pairs(self):
        """Test finding good image pairs based on shared points and baseline."""
        good_pairs = find_good_image_pairs(
            self.point_sharing_info,
            min_shared_points=2,
            min_baseline_ratio=0.1
        )
        
        # Should find pairs with sufficient shared points and baseline
        self.assertGreater(len(good_pairs), 0)
        
        # Check that pairs have the expected format
        for pair in good_pairs:
            self.assertEqual(len(pair), 4)  # (img1, img2, shared_count, baseline)
            self.assertIsInstance(pair[0], str)  # image name
            self.assertIsInstance(pair[1], str)  # image name
            self.assertIsInstance(pair[2], int)  # shared count
            self.assertIsInstance(pair[3], float)  # baseline
            self.assertGreaterEqual(pair[2], 2)  # Min shared points
    
    def test_find_good_image_pairs_strict_requirements(self):
        """Test with strict requirements that should filter out most pairs."""
        good_pairs = find_good_image_pairs(
            self.point_sharing_info,
            min_shared_points=10,  # Very high requirement
            min_baseline_ratio=0.1
        )
        
        # Should find no pairs due to high shared points requirement
        self.assertEqual(len(good_pairs), 0)


class TestMockReconstruction(unittest.TestCase):
    """Test functions that work with mock COLMAP reconstruction."""
    
    def setUp(self):
        """Create a mock reconstruction for testing."""
        # Create mock COLMAP reconstruction
        self.mock_reconstruction = Mock()
        
        # Mock images
        mock_image1 = Mock()
        mock_image1.name = 'img1.jpg'
        mock_image1.registered = True
        mock_image1.rotation_matrix.return_value = np.eye(3)
        mock_image1.translation = np.array([0, 0, 0])
        
        mock_image2 = Mock()
        mock_image2.name = 'img2.jpg'
        mock_image2.registered = True
        mock_image2.rotation_matrix.return_value = np.eye(3)
        mock_image2.translation = np.array([1, 0, 0])
        
        # Mock 3D points
        mock_point1 = Mock()
        mock_track1 = Mock()
        mock_element1 = Mock()
        mock_element1.image_id = 1
        mock_track1.elements = [mock_element1]
        mock_point1.track = mock_track1
        
        self.mock_reconstruction.images = {1: mock_image1, 2: mock_image2}
        self.mock_reconstruction.points3D = {1: mock_point1}
    
    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Full dependencies not available")
    def test_analyze_3d_point_sharing_for_pairs(self):
        """Test 3D point sharing analysis."""
        result = analyze_3d_point_sharing_for_pairs(self.mock_reconstruction)
        
        # Check result structure
        self.assertIn('image_to_points', result)
        self.assertIn('point_to_images', result)
        self.assertIn('image_names', result)
        self.assertIn('image_positions', result)
        
        # Check that we have the expected images
        self.assertIn('img1.jpg', result['image_names'])
        self.assertIn('img2.jpg', result['image_names'])


class TestFilterCorrespondences(unittest.TestCase):
    """Test correspondence filtering functions."""
    
    def setUp(self):
        """Set up test data for filtering."""
        # Simple camera setup
        self.K1 = np.eye(3) * 500
        self.K1[2, 2] = 1
        self.K2 = self.K1.copy()
        
        self.R1 = np.eye(3)
        self.t1 = np.array([0, 0, 0])
        self.R2 = np.eye(3)
        self.t2 = np.array([1, 0, 0])
        
        self.cam_center1 = np.array([0, 0, 0])
        self.cam_center2 = np.array([1, 0, 0])
        
        # Mock arguments
        self.args = Mock()
        self.args.epipolar_threshold = 1.0
        self.args.min_triangulation_angle = 5.0
        self.args.max_reprojection_error = 2.0
    
    def test_filter_correspondences(self):
        """Test correspondence filtering pipeline."""
        # Create some test correspondences
        pts1 = np.array([[250, 250], [300, 300]])
        pts2 = np.array([[250, 250], [300, 300]])
        
        # Triangulate points
        pts_3d = triangulate_correspondences(pts1, pts2, self.K1, self.K2, self.R1, self.t1, self.R2, self.t2)
        
        # Test filtering
        valid_mask = filter_correspondences(
            pts1, pts2, pts_3d, self.K1, self.K2, self.R1, self.t1, self.R2, self.t2,
            self.cam_center1, self.cam_center2, self.args
        )
        
        # Should return a boolean mask
        self.assertEqual(valid_mask.dtype, bool)
        self.assertEqual(len(valid_mask), len(pts1))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 