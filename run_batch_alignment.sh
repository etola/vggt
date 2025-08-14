#!/bin/bash

# Batch alignment script for VGGT reconstruction data
# Aligns batches to the target reconstruction using the new batch-organized structure
#
# Usage:
#   ./run_batch_alignment.sh
#
# This script will:
#   1. Auto-discover available batch directories in the work folder
#   2. Align each batch to the target reconstruction 
#   3. Transform point clouds and save to batch-specific output directory
#   4. Provide detailed progress and summary statistics
#
# Requirements:
#   - Work folder: ~/data/yamac/vggt8/
#   - Batch directories: ~/data/yamac/vggt8/batch_XXX/
#   - Source directories: ~/data/yamac/vggt8/batch_XXX/colmap_calibration/
#   - Point cloud files: ~/data/yamac/vggt8/batch_XXX/ply/combined.ply
#   - Target reconstruction: ~/data/yamac/glomap/sparse/0/
#
# Output:
#   - Transformed reconstructions: ~/data/yamac/vggt8/batch_XXX/transformed/
#   - Progress and validation results for each batch

# Note: We don't use 'set -e' because we want to continue processing other batches 
# even if some individual batches fail

# Configuration
WORK_FOLDER="$HOME/data/yamac/vggt8"
TARGET_DIR="$HOME/data/yamac/glomap/sparse/0/"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting batch alignment process...${NC}"
echo -e "${BLUE}Work folder: ${WORK_FOLDER}${NC}"
echo -e "${BLUE}Target: ${TARGET_DIR}${NC}"
echo -e "${BLUE}Batch structure: batch_XXX/colmap_calibration/, batch_XXX/ply/combined.ply${NC}"
echo ""

# Auto-discover available batches
echo -e "${YELLOW}Auto-discovering available batch directories...${NC}"
batch_dirs=($(find "$WORK_FOLDER" -maxdepth 1 -type d -name "batch_*" | sort))

if [ ${#batch_dirs[@]} -eq 0 ]; then
    echo -e "${RED}ERROR: No batch directories found in $WORK_FOLDER${NC}"
    echo -e "${RED}Expected directories like: batch_000/, batch_001/, etc.${NC}"
    exit 1
fi

echo -e "${GREEN}Found ${#batch_dirs[@]} batch directories:${NC}"
for batch_dir in "${batch_dirs[@]}"; do
    batch_name=$(basename "$batch_dir")
    echo -e "  - $batch_name"
done
echo ""

# Initialize counters
total_batches=${#batch_dirs[@]}
successful=0
failed=0

# Function to run alignment for a single batch
run_alignment() {
    local batch_dir=$1
    local batch_name=$(basename "$batch_dir")
    
    local source_dir="${batch_dir}/colmap_calibration/"
    local ply_file="${batch_dir}/ply/combined.ply"
    local output_dir="${batch_dir}/transformed"
    
    echo -e "${YELLOW}Processing ${batch_name}...${NC}"
    
    # Check if source directory exists
    if [ ! -d "$source_dir" ]; then
        echo -e "${RED}ERROR: Source directory not found: $source_dir${NC}"
        return 1
    fi
    
    # Check if point cloud file exists
    if [ ! -f "$ply_file" ]; then
        echo -e "${RED}ERROR: Point cloud file not found: $ply_file${NC}"
        return 1
    fi
    
    # Ensure output directory exists
    mkdir -p "$output_dir"
    
    # Run the alignment command
    echo "  Command: python align_reconstructions.py -s $source_dir -t $TARGET_DIR -o $output_dir -p $ply_file"
    
    if python align_reconstructions.py \
        -s "$source_dir" \
        -t "$TARGET_DIR" \
        -o "$output_dir" \
        -p "$ply_file"; then
        echo -e "${GREEN}✓ ${batch_name} completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ ${batch_name} failed${NC}"
        return 1
    fi
}

# Main processing loop
echo -e "${BLUE}Processing discovered batches...${NC}"
echo ""

batch_count=0
for batch_dir in "${batch_dirs[@]}"; do
    batch_count=$((batch_count + 1))
    batch_name=$(basename "$batch_dir")
    
    echo -e "${BLUE}=== $batch_name ($batch_count/$total_batches) ===${NC}"
    
    if run_alignment "$batch_dir"; then
        successful=$((successful + 1))
        echo -e "${GREEN}${batch_name} marked as successful. Running total: ${successful}${NC}"
    else
        failed=$((failed + 1))
        echo -e "${RED}${batch_name} marked as failed. Running total: ${failed}${NC}"
    fi
    
    echo -e "${YELLOW}Continuing to next batch...${NC}"
    echo ""
done

# Summary
echo -e "${BLUE}=== BATCH ALIGNMENT SUMMARY ===${NC}"
echo -e "Total batches processed: ${total_batches}"
echo -e "${GREEN}Successful: ${successful}${NC}"
echo -e "${RED}Failed: ${failed}${NC}"

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}🎉 All batches completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some batches failed. Check the output above for details.${NC}"
    exit 1
fi 