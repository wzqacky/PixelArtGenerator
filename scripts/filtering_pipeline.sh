#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_parquet_path> <processed_data_dir>"
    exit 1
fi

INPUT_PARQUET="$1"
PROCESSED_DATA_DIR="$2"

# Output file paths
AESTHETIC_SCORES_CSV="scores_csv/aesthetic_scores.csv"
PIXEL_SCORES_CSV="scores_csv/pixel_scores.csv"
OUTPUT_PARQUET="$PROCESSED_DATA_DIR/$(basename "$INPUT_PARQUET")"

set -e

mkdir -p "scores_csv"
mkdir -p "$PROCESSED_DATA_DIR"

# Step 1: Calculate aesthetic scores
echo "----------------------------------------"
echo "Step 1: Calculating aesthetic scores..."
echo "----------------------------------------"
python filtering/calculate_aesthetic_score.py \
    --dataset_path "$INPUT_PARQUET" \
    --output_file "$AESTHETIC_SCORES_CSV"

# Step 2: Calculate pixel-art similarity scores
echo "------------------------------------------"
echo "Step 2: Calculating pixel-art similarity scores..."
echo "------------------------------------------"
python filtering/calculate_pixel_score.py \
    --dataset_path "$INPUT_PARQUET" \
    --output_file "$PIXEL_SCORES_CSV"

# Step 3: Filter the dataset using both scores
echo "------------------------------------------------"
echo "Step 3: Filtering dataset with both scores..."
echo "------------------------------------------------"
python filtering/filter_dataset.py \
    --input_parquet_path "$INPUT_PARQUET" \
    --csv_files "$AESTHETIC_SCORES_CSV" "$PIXEL_SCORES_CSV" \
    --output_parquet_path "$OUTPUT_PARQUET"

echo "------------------------------------------------"
echo "Filtering complete!"
echo "The final filtered dataset is available at: $OUTPUT_PARQUET"
echo "------------------------------------------------"
