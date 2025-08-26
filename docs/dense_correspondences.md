# Dense Correspondence Extraction

## Overview

The `extract_dense_correspondences.py` script generates dense point clouds by extracting dense features from image pairs using VGGT's DPT head, matching them, and triangulating correspondences. This provides an alternative to depth-based point cloud generation by using feature matching between multiple views.

## Key Features

- **🎯 Smart Pair Selection**: Automatically selects image pairs based on 3D point sharing and baseline requirements from COLMAP reconstruction
- **🔍 Dense Feature Extraction**: Uses VGGT's DPT head to extract dense features for robust matching
- **⚡ Efficient Matching**: Uses reciprocal nearest neighbor matching with subsampling for efficiency
- **🔬 Robust Filtering**: Applies multiple geometric constraints (epipolar, triangulation angle, reprojection error)
- **📐 Coordinate Handling**: Properly handles coordinate transformations between COLMAP full-resolution and VGGT's preprocessed resolution
- **🌐 Automatic Merging**: Combines all pair point clouds into a single dense reconstruction

## Usage

### Basic Usage
```bash
python extract_dense_correspondences.py -s scene/ -g reference_colmap/ -o output/
```

### Quality-Focused (Slower but Better)
```bash
python extract_dense_correspondences.py \
    -s scene/ -g reference_colmap/ -o output/ \
    --subsample_step 2 \
    --max_correspondences 20000 \
    --epipolar_threshold 0.5 \
    --min_triangulation_angle 5.0
```

### Speed-Focused (Faster but Lower Quality)
```bash
python extract_dense_correspondences.py \
    -s scene/ -g reference_colmap/ -o output/ \
    --subsample_step 8 \
    --max_correspondences 5000 \
    --max_pairs 10 \
    --min_baseline_ratio 0.05
```

## Parameters

### Pair Selection
- `--min_shared_points`: Minimum shared 3D points between image pairs (default: 100)
- `--min_baseline_ratio`: Minimum baseline as ratio of scene size (default: 0.1)
- `--max_pairs`: Maximum number of pairs to process (default: None)

### Feature Matching
- `--subsample_step`: Subsampling step for dense features (default: 4)
- `--max_correspondences`: Maximum correspondences per pair (default: 10000)

### Quality Filtering
- `--epipolar_threshold`: Epipolar line distance threshold in pixels (default: 1.0)
- `--min_triangulation_angle`: Minimum triangulation angle in degrees (default: 2.0)
- `--max_reprojection_error`: Maximum reprojection error in pixels (default: 4.0)

## Output Structure

```
output_dir/
├── dense_pointcloud.ply          # 🌐 Final merged point cloud (main result)
├── processing_summary.json       # 📊 Overall statistics and results
├── pair_000/                     # Individual pair results
│   ├── pointcloud.ply            # Point cloud for this pair
│   └── metadata.json             # Pair-specific metadata
├── pair_001/
│   ├── pointcloud.ply
│   └── metadata.json
└── ...
```

## Algorithm Overview

1. **📊 Analysis**: Analyze 3D point sharing between images in reference COLMAP reconstruction
2. **🎯 Pair Selection**: Find image pairs with sufficient shared points and baseline
3. **🔍 Feature Extraction**: Extract dense features using VGGT's DPT head
4. **⚡ Matching**: Match features using reciprocal nearest neighbors
5. **📐 Triangulation**: Triangulate correspondences to generate 3D points
6. **🔬 Filtering**: Apply geometric filters to remove erroneous points
7. **🎨 Coloring**: Sample colors from original images
8. **🌐 Merging**: Combine all pair point clouds into final result

## Coordinate System Handling

The script properly handles the coordinate transformation between:
- **COLMAP coordinates**: Full-resolution image coordinates used in the reference calibration
- **VGGT coordinates**: Model resolution (518×518) with padding applied during preprocessing

The intrinsic camera matrices are automatically adjusted to account for the padding and scaling applied by VGGT's preprocessing.

## Tips for Best Results

1. **Start with defaults**: The default parameters work well for most scenes
2. **Quality vs Speed**: 
   - For quality: decrease `--subsample_step`, increase filtering thresholds
   - For speed: increase `--subsample_step`, decrease filtering thresholds
3. **Large datasets**: Use `--max_pairs` to limit processing time
4. **Debugging**: Use `--max_pairs 3` with loose constraints for quick testing
5. **Monitor success**: Check `processing_summary.json` for pair success rates

## Example Usage Script

Run the interactive examples:
```bash
python example_dense_correspondences.py
```

This provides pre-configured examples for different use cases and explains parameter tuning.

## Testing

Unit tests are available in `tests/test_dense_correspondences.py`:
```bash
python tests/test_dense_correspondences.py
```

Tests cover:
- Camera intrinsics adjustment for VGGT preprocessing
- Geometric functions (triangulation, epipolar distance, etc.)
- Image pair selection logic
- Correspondence filtering pipeline

## Dependencies

- VGGT model with DPT head
- NumPy, OpenCV
- Trimesh for point cloud I/O
- PyTorch for VGGT inference
- COLMAP reconstruction utilities

## Comparison with Depth-Based Methods

| Aspect | Dense Correspondences | Depth-Based (generate_batched_pointcloud.py) |
|--------|----------------------|----------------------------------------------|
| **Input** | Image pairs | Individual images |
| **Method** | Feature matching + triangulation | Depth estimation + unprojection |
| **Accuracy** | Multi-view constraints | Single-view estimation |
| **Speed** | Slower (pair processing) | Faster (individual images) |
| **Coverage** | Depends on matches | Full image coverage |
| **Reliability** | High (geometric verification) | Moderate (depth estimation) |

Use dense correspondences when you need high accuracy with geometric verification, and depth-based methods when you need fast processing or full coverage. 