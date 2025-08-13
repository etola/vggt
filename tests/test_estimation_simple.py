#!/usr/bin/env python3
import numpy as np
import sys
import os

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define estimation functions locally to avoid import issues
def _pairwise_distance_ratios(points_src: np.ndarray, points_dst: np.ndarray) -> np.ndarray:
    """Compute ratios of pairwise distances ||p_i - p_j||_dst / ||p_i - p_j||_src for all i<j."""
    n = points_src.shape[0]
    if n < 2:
        raise ValueError("Need at least two points to compute pairwise distance ratios.")
    ratios = []
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
    """Estimate scale s such that s * points_src ~ points_dst (both in same frame up to rigid)."""
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


def estimate_rigid_transform(points_src: np.ndarray, points_dst: np.ndarray):
    """Estimate rotation R (3x3) and translation t (3,) such that R * p + t ≈ q."""
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


def apply_similarity_transform_to_point(point: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply similarity transform to a single 3D point."""
    point = np.asarray(point)
    if point.shape != (3,):
        raise ValueError(f"Point must be a 3D vector, got shape {point.shape}")
    return scale * (rotation @ point) + translation


class SimpleEstimationTests:
    """Simple tests for similarity transform estimation."""
    
    def __init__(self):
        self.tolerance = 1e-10
        self.loose_tolerance = 1e-6
        
        # Create synthetic test data
        np.random.seed(42)  # For reproducible tests
        self.n_points = 6
        self.src_points = np.random.randn(self.n_points, 3) * 2.0
        
        # Known transform parameters
        self.true_scale = 2.5
        self.true_rotation = self._create_rotation_matrix(30, 45, 60)
        self.true_translation = np.array([1.5, -2.0, 3.5])
        
        # Apply known transform
        self.dst_points = self.true_scale * (self.true_rotation @ self.src_points.T).T + self.true_translation
    
    def _create_rotation_matrix(self, rx_deg, ry_deg, rz_deg):
        """Create rotation matrix from Euler angles in degrees."""
        rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
        
        # Rotation matrices
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
    
    def test_pairwise_distance_ratios(self):
        """Test pairwise distance ratio computation."""
        print("Testing pairwise distance ratios...")
        
        # Simple case: uniform scaling
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        scale = 3.7
        dst = scale * src
        
        ratios = _pairwise_distance_ratios(src, dst)
        
        # All ratios should equal the scale factor
        expected_ratios = np.full(len(ratios), scale)
        if not np.allclose(ratios, expected_ratios, atol=self.tolerance):
            print(f"✗ Failed: ratios {ratios} != expected {expected_ratios}")
            return False
        
        # Test median equals scale
        if not abs(np.median(ratios) - scale) < self.tolerance:
            print(f"✗ Failed: median {np.median(ratios)} != scale {scale}")
            return False
        
        print("✓ Pairwise distance ratios test passed")
        return True
    
    def test_scale_estimation_two_points(self):
        """Test scale estimation with exactly two points."""
        print("Testing scale estimation with two points...")
        
        src_two = self.src_points[:2]
        dst_two = self.dst_points[:2]
        
        # Test both methods
        for robust in [True, False]:
            estimated_scale = estimate_scale_from_centers(src_two, dst_two, robust=robust)
            if not abs(estimated_scale - self.true_scale) < self.loose_tolerance:
                print(f"✗ Failed (robust={robust}): estimated scale {estimated_scale} != true scale {self.true_scale}")
                return False
        
        print("✓ Scale estimation (two points) test passed")
        return True
    
    def test_scale_estimation_multiple_points(self):
        """Test scale estimation with multiple points."""
        print("Testing scale estimation with multiple points...")
        
        for robust in [True, False]:
            estimated_scale = estimate_scale_from_centers(self.src_points, self.dst_points, robust=robust)
            if not abs(estimated_scale - self.true_scale) < self.loose_tolerance:
                print(f"✗ Failed (robust={robust}): estimated scale {estimated_scale} != true scale {self.true_scale}")
                return False
        
        print("✓ Scale estimation (multiple points) test passed")
        return True
    
    def test_rigid_transform_estimation(self):
        """Test rigid transform estimation."""
        print("Testing rigid transform estimation...")
        
        # Use scaled source points (scale already applied)
        scaled_src = self.true_scale * self.src_points
        
        estimated_R, estimated_t = estimate_rigid_transform(scaled_src, self.dst_points)
        
        # Verify the transform works
        transformed = (estimated_R @ scaled_src.T).T + estimated_t
        if not np.allclose(transformed, self.dst_points, atol=self.loose_tolerance):
            print(f"✗ Failed: transform doesn't align points correctly")
            return False
        
        # Check orthogonality: R @ R.T = I
        should_be_identity = estimated_R @ estimated_R.T
        if not np.allclose(should_be_identity, np.eye(3), atol=self.loose_tolerance):
            print(f"✗ Failed: rotation matrix not orthogonal")
            return False
        
        # Check determinant is +1 (proper rotation)
        det = np.linalg.det(estimated_R)
        if not abs(det - 1.0) < self.loose_tolerance:
            print(f"✗ Failed: determinant {det} != 1.0")
            return False
        
        print("✓ Rigid transform estimation test passed")
        return True
    
    def test_end_to_end_estimation(self):
        """Test end-to-end similarity transform estimation."""
        print("Testing end-to-end similarity transform estimation...")
        
        # Step 1: Estimate scale
        estimated_scale = estimate_scale_from_centers(self.src_points, self.dst_points, robust=True)
        
        # Step 2: Scale source points and estimate rigid transform
        scaled_src = estimated_scale * self.src_points
        estimated_R, estimated_t = estimate_rigid_transform(scaled_src, self.dst_points)
        
        # Step 3: Verify parameters are close to ground truth
        if not abs(estimated_scale - self.true_scale) < self.loose_tolerance:
            print(f"✗ Failed: estimated scale {estimated_scale} != true scale {self.true_scale}")
            return False
        
        if not np.allclose(estimated_R, self.true_rotation, atol=self.loose_tolerance):
            print(f"✗ Failed: estimated rotation differs from true rotation")
            return False
        
        if not np.allclose(estimated_t, self.true_translation, atol=self.loose_tolerance):
            print(f"✗ Failed: estimated translation differs from true translation")
            return False
        
        # Step 4: Verify transform works by applying to all points
        for i, src_point in enumerate(self.src_points):
            transformed = apply_similarity_transform_to_point(
                src_point, estimated_scale, estimated_R, estimated_t
            )
            if not np.allclose(transformed, self.dst_points[i], atol=self.loose_tolerance):
                print(f"✗ Failed: point {i} transform incorrect")
                return False
        
        print("✓ End-to-end estimation test passed")
        return True
    
    def test_minimal_case_two_points(self):
        """Test estimation with minimal case (2 points)."""
        print("Testing minimal case (2 points)...")
        
        src_two = self.src_points[:2]
        dst_two = self.dst_points[:2]
        
        # Estimate transform
        estimated_scale = estimate_scale_from_centers(src_two, dst_two, robust=True)
        scaled_src = estimated_scale * src_two
        estimated_R, estimated_t = estimate_rigid_transform(scaled_src, dst_two)
        
        # Verify it perfectly aligns the two points
        transformed = estimated_scale * (estimated_R @ src_two.T).T + estimated_t
        if not np.allclose(transformed, dst_two, atol=self.tolerance):
            print(f"✗ Failed: minimal case doesn't align points perfectly")
            return False
        
        print("✓ Minimal case (2 points) test passed")
        return True
    
    def test_special_cases(self):
        """Test special transform cases."""
        print("Testing special cases...")
        
        # Test identity transform
        src = self.src_points
        dst = self.src_points.copy()
        
        estimated_scale = estimate_scale_from_centers(src, dst, robust=True)
        scaled_src = estimated_scale * src
        estimated_R, estimated_t = estimate_rigid_transform(scaled_src, dst)
        
        # Should recover identity transform
        if not abs(estimated_scale - 1.0) < self.loose_tolerance:
            print(f"✗ Failed identity: scale {estimated_scale} != 1.0")
            return False
        
        if not np.allclose(estimated_R, np.eye(3), atol=self.loose_tolerance):
            print(f"✗ Failed identity: rotation not identity")
            return False
        
        if not np.allclose(estimated_t, np.zeros(3), atol=self.loose_tolerance):
            print(f"✗ Failed identity: translation not zero")
            return False
        
        # Test pure scale
        scale_only = 4.2
        dst_scale = scale_only * src
        
        estimated_scale = estimate_scale_from_centers(src, dst_scale, robust=True)
        if not abs(estimated_scale - scale_only) < self.loose_tolerance:
            print(f"✗ Failed pure scale: {estimated_scale} != {scale_only}")
            return False
        
        print("✓ Special cases test passed")
        return True
    
    def test_error_handling(self):
        """Test error handling for degenerate cases."""
        print("Testing error handling...")
        
        # Single point should fail
        try:
            estimate_scale_from_centers(np.array([[1, 2, 3]]), np.array([[2, 4, 6]]))
            print("✗ Failed: should have raised ValueError for single point")
            return False
        except ValueError:
            pass  # Expected
        
        # Single point for rigid transform should fail
        try:
            estimate_rigid_transform(np.array([[1, 2, 3]]), np.array([[2, 4, 6]]))
            print("✗ Failed: should have raised ValueError for single point in rigid transform")
            return False
        except ValueError:
            pass  # Expected
        
        print("✓ Error handling test passed")
        return True


def main():
    """Run all simple estimation tests."""
    print("Running simple similarity transform estimation tests...")
    print("=" * 70)
    
    tester = SimpleEstimationTests()
    
    tests = [
        tester.test_pairwise_distance_ratios,
        tester.test_scale_estimation_two_points,
        tester.test_scale_estimation_multiple_points,
        tester.test_rigid_transform_estimation,
        tester.test_end_to_end_estimation,
        tester.test_minimal_case_two_points,
        tester.test_special_cases,
        tester.test_error_handling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 70)
    print(f"Estimation Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All estimation tests passed!")
        return 0
    else:
        print("❌ Some estimation tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 