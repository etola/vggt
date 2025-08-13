# Unit Tests for Reconstruction Transform

This directory contains unit tests for the reconstruction transform functionality.

## Test Files

### `test_direct_transform.py`
- **Recommended for general use**
- Standalone test that doesn't require external dependencies beyond numpy
- Tests the `apply_similarity_transform_to_point` function with comprehensive test cases
- Can be run directly: `python3 tests/test_direct_transform.py`

### `test_reconstruction_transform.py`
- Comprehensive unit tests using unittest framework
- Requires scipy for full functionality
- Tests multiple functions including scale estimation and rigid transforms
- Run with: `python3 -m unittest tests.test_reconstruction_transform` (when dependencies available)

### `test_simple_transform.py`
- Alternative simple test (may have import issues depending on environment)
- Kept for reference and debugging

### `test_similarity_estimation.py`
- Comprehensive unit tests for similarity transform estimation functions
- Requires scipy and pycolmap for full functionality
- Tests scale estimation, rigid transform estimation, and end-to-end pipeline
- Includes synthetic COLMAP reconstruction tests

### `test_estimation_simple.py`
- **Recommended for estimation testing**
- Standalone test for similarity transform estimation requiring only numpy
- Tests pairwise distance ratios, scale estimation, rigid transforms
- Covers edge cases, error handling, and end-to-end estimation pipeline

### `test_compatibility_functions.py`
- **Tests for compatibility wrapper functions**
- Tests functions added for backward compatibility with legacy code
- Verifies batch point transformation, point cloud transformation
- Tests wrapper functions that maintain old API signatures

### `test_pointcloud_alignment.py`
- **Tests for point cloud alignment feature in align_reconstructions.py**
- Tests end-to-end point cloud transformation with synthetic data
- Verifies point cloud transformation accuracy and output handling
- Tests default output path generation

## Running Tests

### Quick Tests (No dependencies needed)
```bash
cd /home/tola/code/reference/vggt

# Test point transformation functions
python3 tests/test_direct_transform.py

# Test similarity transform estimation functions  
python3 tests/test_estimation_simple.py

# Test compatibility wrapper functions
python3 tests/test_compatibility_functions.py

# Test point cloud alignment feature
python3 tests/test_pointcloud_alignment.py
```

### Full Test Suite (Requires scipy, pycolmap)
```bash
cd /home/tola/code/reference/vggt

# Full point transformation tests
python3 -m unittest tests.test_reconstruction_transform

# Full estimation tests with synthetic COLMAP data
python3 -m unittest tests.test_similarity_estimation

# Full compatibility function tests
python3 tests/test_compatibility_functions.py
```

## Test Coverage

The tests verify:

**Point Transformation:**
- Identity transforms
- Pure scaling, translation, rotation operations  
- Combined similarity transforms
- Input validation and error handling
- Consistency between individual and batch transforms
- Manual calculation verification

**Similarity Transform Estimation:**
- Pairwise distance ratio computation
- Scale estimation (robust and non-robust methods)
- Rigid transform estimation (rotation + translation)
- End-to-end estimation pipeline
- Minimal case handling (2 cameras)
- Special cases (identity, pure scale/translation)
- Error handling for degenerate configurations
- Synthetic COLMAP reconstruction alignment

**Compatibility Functions:**
- Batch point transformation (multiple points at once)
- Point cloud transformation with color preservation
- Wrapper functions maintaining legacy API compatibility
- Empty array handling and edge cases
- Function signature verification

**Point Cloud Alignment:**
- End-to-end CLI testing with synthetic COLMAP reconstructions
- Point cloud transformation accuracy verification
- Color preservation during transformation
- Default output path generation and handling
- Integration testing with trimesh library

All tests use high precision (1e-10 tolerance) to ensure numerical accuracy. 