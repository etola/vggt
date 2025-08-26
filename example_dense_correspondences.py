#!/usr/bin/env python3
"""
Example usage of the dense correspondence extraction script.

This script demonstrates how to use extract_dense_correspondences.py
with different parameter configurations for various use cases.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Command completed successfully!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with error code {e.returncode}")
        if e.stdout:
            print("stdout:")
            print(e.stdout)
        if e.stderr:
            print("stderr:")
            print(e.stderr)
        return False


def check_paths(scene_dir, reference_calibration):
    """Check if required paths exist."""
    scene_path = Path(scene_dir)
    ref_path = Path(reference_calibration)
    
    if not scene_path.exists():
        print(f"❌ Scene directory not found: {scene_dir}")
        return False
    
    images_dir = scene_path / "images"
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    if not ref_path.exists():
        print(f"❌ Reference calibration not found: {reference_calibration}")
        return False
    
    # Check for required COLMAP files
    required_files = ["cameras.txt", "images.txt", "points3D.txt"]
    for file in required_files:
        if not (ref_path / file).exists():
            print(f"❌ Required COLMAP file not found: {ref_path / file}")
            return False
    
    print("✅ All required paths and files found")
    return True


def example_basic_usage():
    """Basic usage example with default parameters."""
    print("""
📋 EXAMPLE 1: Basic Usage with Default Parameters
=================================================

This is the simplest way to run dense correspondence extraction.
Uses default parameters which work well for most scenes.
""")
    
    # Example paths (adjust these for your data)
    scene_dir = "examples/kitchen"
    reference_calibration = "path/to/reference_colmap"
    output_dir = "output/dense_correspondences_basic"
    
    if not check_paths(scene_dir, reference_calibration):
        print("⚠️  Adjust the paths in this example to match your data")
        return
    
    cmd = [
        "python", "extract_dense_correspondences.py",
        "-s", scene_dir,
        "-g", reference_calibration,
        "-o", output_dir
    ]
    
    run_command(cmd, "Basic Dense Correspondence Extraction")


def example_quality_focused():
    """Example with parameters focused on quality over speed."""
    print("""
📋 EXAMPLE 2: Quality-Focused Parameters
========================================

This configuration prioritizes quality over speed:
- Lower subsampling step for denser features
- Stricter filtering thresholds
- More correspondences per pair
""")
    
    scene_dir = "examples/kitchen"
    reference_calibration = "path/to/reference_colmap"
    output_dir = "output/dense_correspondences_quality"
    
    cmd = [
        "python", "extract_dense_correspondences.py",
        "-s", scene_dir,
        "-g", reference_calibration,
        "-o", output_dir,
        "--subsample_step", "2",  # Denser feature sampling
        "--max_correspondences", "20000",  # More correspondences
        "--epipolar_threshold", "0.5",  # Stricter epipolar constraint
        "--min_triangulation_angle", "5.0",  # Larger triangulation angle
        "--max_reprojection_error", "2.0",  # Lower reprojection error
        "--min_baseline_ratio", "0.15"  # Larger baseline requirement
    ]
    
    run_command(cmd, "Quality-Focused Dense Correspondence Extraction")


def example_speed_focused():
    """Example with parameters focused on speed over quality."""
    print("""
📋 EXAMPLE 3: Speed-Focused Parameters
======================================

This configuration prioritizes speed over quality:
- Higher subsampling step for fewer features
- Looser filtering thresholds
- Fewer correspondences per pair
- Limit number of pairs processed
""")
    
    scene_dir = "examples/kitchen"
    reference_calibration = "path/to/reference_colmap"
    output_dir = "output/dense_correspondences_speed"
    
    cmd = [
        "python", "extract_dense_correspondences.py",
        "-s", scene_dir,
        "-g", reference_calibration,
        "-o", output_dir,
        "--subsample_step", "8",  # Coarser feature sampling
        "--max_correspondences", "5000",  # Fewer correspondences
        "--max_pairs", "10",  # Limit number of pairs
        "--epipolar_threshold", "2.0",  # Looser epipolar constraint
        "--min_triangulation_angle", "1.0",  # Smaller triangulation angle
        "--max_reprojection_error", "8.0",  # Higher reprojection error
        "--min_baseline_ratio", "0.05"  # Smaller baseline requirement
    ]
    
    run_command(cmd, "Speed-Focused Dense Correspondence Extraction")


def example_selective_pairs():
    """Example with selective pair processing."""
    print("""
📋 EXAMPLE 4: Selective Pair Processing
=======================================

This configuration is useful for large datasets where you want
to process only the best image pairs:
- High shared point requirement
- Large baseline requirement
- Limited number of pairs
""")
    
    scene_dir = "examples/kitchen"
    reference_calibration = "path/to/reference_colmap"
    output_dir = "output/dense_correspondences_selective"
    
    cmd = [
        "python", "extract_dense_correspondences.py",
        "-s", scene_dir,
        "-g", reference_calibration,
        "-o", output_dir,
        "--min_shared_points", "200",  # High shared point requirement
        "--min_baseline_ratio", "0.2",  # Large baseline requirement
        "--max_pairs", "20",  # Process only best pairs
        "--subsample_step", "4",
        "--max_correspondences", "15000"
    ]
    
    run_command(cmd, "Selective Pair Processing")


def example_debugging():
    """Example with parameters useful for debugging."""
    print("""
📋 EXAMPLE 5: Debugging Configuration
=====================================

This configuration is useful for debugging and understanding
what the algorithm is doing:
- Process only a few pairs
- Looser constraints to see more points
- Smaller scene for faster iteration
""")
    
    scene_dir = "examples/kitchen"
    reference_calibration = "path/to/reference_colmap"
    output_dir = "output/dense_correspondences_debug"
    
    cmd = [
        "python", "extract_dense_correspondences.py",
        "-s", scene_dir,
        "-g", reference_calibration,
        "-o", output_dir,
        "--max_pairs", "3",  # Just a few pairs for debugging
        "--min_shared_points", "50",  # Lower requirement
        "--min_baseline_ratio", "0.05",  # Lower baseline requirement
        "--subsample_step", "6",
        "--max_correspondences", "5000",
        "--seed", "42"  # Fixed seed for reproducible results
    ]
    
    run_command(cmd, "Debugging Configuration")


def show_output_structure():
    """Show the expected output structure."""
    print("""
📁 OUTPUT STRUCTURE
==================

After running the dense correspondence extraction, you'll get:

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

Key files:
- dense_pointcloud.ply: The main result - merged point cloud from all pairs
- processing_summary.json: Contains statistics about success rates, point counts, etc.
- pair_xxx/pointcloud.ply: Individual point clouds for each image pair
- pair_xxx/metadata.json: Detailed information about each pair's processing

Visualization:
You can view the point clouds using:
- CloudCompare
- MeshLab  
- Open3D
- PCL viewer
- Or any PLY file viewer
""")


def show_parameter_guide():
    """Show parameter tuning guide."""
    print("""
🔧 PARAMETER TUNING GUIDE
=========================

Key parameters and their effects:

🎯 Pair Selection:
--min_shared_points N     Higher = more reliable pairs, fewer pairs
--min_baseline_ratio F    Higher = better triangulation, fewer pairs  
--max_pairs N            Limits processing time

🔍 Feature Matching:
--subsample_step N       Lower = denser features, slower processing
--max_correspondences N  Higher = more points, slower processing

🔬 Quality Filtering:
--epipolar_threshold F   Lower = stricter geometry, fewer points
--min_triangulation_angle F  Higher = better triangulation, fewer points
--max_reprojection_error F   Lower = more accurate points, fewer points

💡 Tips:
1. Start with default parameters
2. For quality: decrease subsample_step, increase filtering thresholds
3. For speed: increase subsample_step, decrease filtering thresholds
4. For debugging: use --max_pairs 3 and loose constraints
5. Monitor the processing_summary.json for success rates
""")


def main():
    """Main example runner."""
    print("""
🌟 VGGT Dense Correspondence Extraction Examples
================================================

This script shows different ways to use the dense correspondence 
extraction tool for various scenarios.

Choose an example to run:
""")
    
    examples = [
        ("1", "Basic usage with default parameters", example_basic_usage),
        ("2", "Quality-focused configuration", example_quality_focused),
        ("3", "Speed-focused configuration", example_speed_focused),
        ("4", "Selective pair processing", example_selective_pairs),
        ("5", "Debugging configuration", example_debugging),
        ("s", "Show output structure", show_output_structure),
        ("p", "Show parameter tuning guide", show_parameter_guide),
        ("a", "Run all examples (non-interactive)", lambda: run_all_examples())
    ]
    
    for key, desc, _ in examples:
        print(f"  {key}) {desc}")
    
    print("\nEnter your choice (or 'q' to quit): ", end="")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
    else:
        choice = input().strip().lower()
    
    if choice == 'q':
        return
    
    for key, desc, func in examples:
        if choice == key:
            func()
            return
    
    print("Invalid choice!")


def run_all_examples():
    """Run all examples in sequence."""
    print("Running all examples...")
    example_basic_usage()
    example_quality_focused()
    example_speed_focused()
    example_selective_pairs()
    example_debugging()
    show_output_structure()
    show_parameter_guide()


if __name__ == "__main__":
    main() 