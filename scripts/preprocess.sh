#!/bin/bash
set -e -i

REPO_ID="bghira/free-to-use-pixelart"
RAW_DATA_DIR="data/pixilart"
PROCESSED_DATA_DIR="data/pixilart_processed"

# Download the raw datasets
echo ">>> Step 1: Downloading raw dataset (${REPO_ID}) from HuggingFace and KaggleHub..."
python preprocess/download_hf.py --repo_id bghira/free-to-use-pixelart --local_dir ${RAW_DATA_DIR} --repo_type dataset
python preprocess/download_kaggle.py --data_id artvandaley/curated-pixel-art-512x512 --local_dir data/pixel-art-512x512
echo "Download complete."


echo ">>> Processing Parquet file and downloading images to ${PROCESSED_DATA_DIR}..."
PARQUET_FILE_PATH=$(find "${RAW_DATA_DIR}" -type f -name "*.parquet" | head -n 1)

if [ -z "${PARQUET_FILE_PATH}" ]; then
    echo "Error: No .parquet file found in the downloaded dataset at ${RAW_DATA_DIR}."
    exit 1
fi
echo "Found parquet file: ${PARQUET_FILE_PATH}"


python preprocess/process_parquet.py --file_path "${PARQUET_FILE_PATH}" --output_dir "${PROCESSED_DATA_DIR}"
echo "--- Preprocessing complete. ---"
