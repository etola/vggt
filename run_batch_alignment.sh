#!/bin/bash

# Batch alignment script for VGGT2 reconstruction data
# Aligns batches 000 through 033 to the target reconstruction
#
# Usage:
#   ./run_batch_alignment.sh
#
# This script will:
#   1. Process batches 000 through 033
#   2. Align each batch to the target reconstruction 
#   3. Transform point clouds and save to output directory
#   4. Provide detailed progress and summary statistics
#
# Requirements:
#   - Source directories: ~/data/yamac/vggt2/colmap_calibration/batch_XXX/
#   - Point cloud files: ~/data/yamac/vggt2/ply/batch_XXX_combined.ply
#   - Target reconstruction: ~/data/yamac/glomap/sparse/0/
#
# Output:
#   - Transformed reconstructions: ~/data/yamac/vggt2/transformed/
#   - Aligned point clouds: ~/data/yamac/vggt2/transformed/aligned_XXX.ply
#   - Validation results for each batch

# Note: We don't use 'set -e' because we want to continue processing other batches 
# even if some individual batches fail

# Configuration
SOURCE_BASE="$HOME/data/yamac/vggt8/colmap_calibration/batch_"
TARGET_DIR="$HOME/data/yamac/glomap/sparse/0/"
OUTPUT_DIR="$HOME/data/yamac/vggt8/transformed"
PLY_BASE="$HOME/data/yamac/vggt8/ply/batch_"
PLY_SUFFIX="_combined.ply"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting batch alignment process...${NC}"
echo -e "${BLUE}Source base: ${SOURCE_BASE}XXX/${NC}"
echo -e "${BLUE}Target: ${TARGET_DIR}${NC}"
echo -e "${BLUE}Output: ${OUTPUT_DIR}${NC}"
echo -e "${BLUE}Point cloud base: ${PLY_BASE}XXX${PLY_SUFFIX}${NC}"
echo ""

# Initialize counters
total_batches=34  # 000 to 033 inclusive
successful=0
failed=0

# Function to run alignment for a single batch
run_alignment() {
    local batch_num=$1
    local batch_id=$(printf "%03d" $batch_num)
    
    local source_dir="${SOURCE_BASE}${batch_id}/"
    local ply_file="${PLY_BASE}${batch_id}${PLY_SUFFIX}"
    local output_ply="aligned_${batch_id}.ply"
    
    echo -e "${YELLOW}Processing batch ${batch_id}...${NC}"
    
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
    
    # Run the alignment command
    echo "  Command: python align_reconstructions.py -s $source_dir -t $TARGET_DIR -o $OUTPUT_DIR -p $ply_file -po $output_ply"
    
    if python align_reconstructions.py \
        -s "$source_dir" \
        -t "$TARGET_DIR" \
        -o "$OUTPUT_DIR" \
        -p "$ply_file" \
        -po "$output_ply"; then
        echo -e "${GREEN}✓ Batch ${batch_id} completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ Batch ${batch_id} failed${NC}"
        return 1
    fi
}

# Main processing loop
echo -e "${BLUE}Processing batches 000 through 033...${NC}"
echo ""

for i in {0..33}; do
    batch_id=$(printf "%03d" $i)
    echo -e "${BLUE}=== BATCH ${batch_id} ($(($i + 1))/${total_batches}) ===${NC}"
    
    if run_alignment $i; then
        successful=$((successful + 1))
        echo -e "${GREEN}Batch ${batch_id} marked as successful. Running total: ${successful}${NC}"
    else
        failed=$((failed + 1))
        echo -e "${RED}Batch ${batch_id} marked as failed. Running total: ${failed}${NC}"
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