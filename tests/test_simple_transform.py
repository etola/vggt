#!/usr/bin/env python3
import numpy as np
import sys
import os

# Add current directory to path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.reconstruction_transform import apply_similarity_transform_to_point


def test_identity_transform():
    """Test identity transform."""
    print("Testing identity transform...")
    point = np.array([1.0, 2.0, 3.0])
    scale = 1.0
    rotation = np.eye(3)
    translation = np.zeros(3)
    
    result = apply_similarity_transform_to_point(point, scale, rotation, translation)
    expected = point
    
    if np.allclose(result, expected, atol=1e-10):
        print("✓ Identity transform test passed")
        return True
    else:
        print(f"✗ Identity transform test failed: {result} != {expected}")
        return False


def test_scale_only():
    """Test pure scaling."""
    print("Testing scale-only transform...")
    point = np.array([1.0, 2.0, 3.0])
    scale = 2.5
    rotation = np.eye(3)
    translation = np.zeros(3)
    
    result = apply_similarity_transform_to_point(point, scale, rotation, translation)
    expected = scale * point
    
    if np.allclose(result, expected, atol=1e-10):
        print("✓ Scale-only transform test passed")
        return True
    else:
        print(f"✗ Scale-only transform test failed: {result} != {expected}")
        return False


def test_translation_only():
    """Test pure translation."""
    print("Testing translation-only transform...")
    point = np.array([1.0, 2.0, 3.0])
    scale = 1.0
    rotation = np.eye(3)
    translation = np.array([5.0, -3.0, 1.5])
    
    result = apply_similarity_transform_to_point(point, scale, rotation, translation)
    expected = point + translation
    
    if np.allclose(result, expected, atol=1e-10):
        print("✓ Translation-only transform test passed")
        return True
    else:
        print(f"✗ Translation-only transform test failed: {result} != {expected}")
        return False


def test_rotation_90z():
    """Test 90-degree rotation around Z axis."""
    print("Testing 90° rotation around Z axis...")
    point = np.array([1.0, 0.0, 0.0])
    scale = 1.0
    # 90 degree rotation around Z axis matrix
    rotation = np.array([[0.0, -1.0, 0.0],
                        [1.0,  0.0, 0.0],
                        [0.0,  0.0, 1.0]])
    translation = np.zeros(3)
    
    result = apply_similarity_transform_to_point(point, scale, rotation, translation)
    expected = np.array([0.0, 1.0, 0.0])
    
    if np.allclose(result, expected, atol=1e-10):
        print("✓ 90° Z rotation test passed")
        return True
    else:
        print(f"✗ 90° Z rotation test failed: {result} != {expected}")
        return False


def test_combined_transform():
    """Test combined scale, rotation, and translation."""
    print("Testing combined transform...")
    point = np.array([1.0, 0.0, 0.0])
    scale = 2.0
    # 90 degree rotation around Z axis
    rotation = np.array([[0.0, -1.0, 0.0],
                        [1.0,  0.0, 0.0],
                        [0.0,  0.0, 1.0]])
    translation = np.array([1.0, 1.0, 1.0])
    
    result = apply_similarity_transform_to_point(point, scale, rotation, translation)
    # Expected: scale * rotate * point + translation = 2.0 * [0,1,0] + [1,1,1] = [1,3,1]
    expected = np.array([1.0, 3.0, 1.0])
    
    if np.allclose(result, expected, atol=1e-10):
        print("✓ Combined transform test passed")
        return True
    else:
        print(f"✗ Combined transform test failed: {result} != {expected}")
        return False


def test_invalid_input():
    """Test error handling for invalid input."""
    print("Testing invalid input handling...")
    scale = 1.0
    rotation = np.eye(3)
    translation = np.zeros(3)
    
    try:
        # Test wrong shape - 2D point
        apply_similarity_transform_to_point(np.array([1.0, 2.0]), scale, rotation, translation)
        print("✗ Invalid input test failed: should have raised ValueError for 2D point")
        return False
    except ValueError:
        pass  # Expected
    
    try:
        # Test wrong shape - matrix instead of vector
        apply_similarity_transform_to_point(np.array([[1.0, 2.0, 3.0]]), scale, rotation, translation)
        print("✗ Invalid input test failed: should have raised ValueError for matrix input")
        return False
    except ValueError:
        pass  # Expected
    
    print("✓ Invalid input handling test passed")
    return True


def test_manual_vs_function_consistency():
    """Test that function gives same result as manual calculation."""
    print("Testing manual vs function consistency...")
    point = np.array([1.5, -2.3, 0.8])
    scale = 1.7
    # Simple rotation matrix (45 degrees around Z)
    angle = np.pi / 4  # 45 degrees
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle),  np.cos(angle), 0],
                        [0,              0,             1]])
    translation = np.array([-0.5, 2.1, -1.2])
    
    # Apply using function
    result_function = apply_similarity_transform_to_point(point, scale, rotation, translation)
    
    # Apply manually
    result_manual = scale * (rotation @ point) + translation
    
    if np.allclose(result_function, result_manual, atol=1e-10):
        print("✓ Manual vs function consistency test passed")
        return True
    else:
        print(f"✗ Manual vs function consistency test failed:")
        print(f"  Function result: {result_function}")
        print(f"  Manual result:   {result_manual}")
        return False


def main():
    """Run all tests."""
    print("Running simple transform tests...")
    print("=" * 50)
    
    tests = [
        test_identity_transform,
        test_scale_only,
        test_translation_only,
        test_rotation_90z,
        test_combined_transform,
        test_invalid_input,
        test_manual_vs_function_consistency,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 