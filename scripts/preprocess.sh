#!/bin/bash
set -e -i

REPO_ID="bghira/free-to-use-pixelart"
RAW_DATA_DIR_HF="data/pixilart"
RAW_DATA_DIR_KAGGLE="data/pixel-art-512x512"
PROCESSED_DATA_DIR="data/pixilart_processed"

# Download the raw datasets
echo ">>> Step 1: Downloading raw dataset (${REPO_ID}) from HuggingFace and KaggleHub..."
if [ -d "${RAW_DATA_DIR_HF}" ]; then
    echo "HuggingFace dataset already exists."
else
    python preprocess/download_hf.py --repo_id bghira/free-to-use-pixelart --local_dir ${RAW_DATA_DIR_HF} --repo_type dataset
fi
if [ -d "${RAW_DATA_DIR_KAGGLE} "]; then
    echo "Kaggle dataset already exists."
else
    python preprocess/download_kaggle.py --data_id artvandaley/curated-pixel-art-512x512 --local_dir ${RAW_DATA_DIR_KAGGLE}
fi
echo "Download complete."

echo ">>> Processing Parquet file and downloading images to ${PROCESSED_DATA_DIR}..."
PARQUET_FILE_PATH=$(find "${RAW_DATA_DIR_HF}" -type f -name "*.parquet" | head -n 1)

if [ -z "${PARQUET_FILE_PATH}" ]; then
    echo "Error: No .parquet file found in the downloaded dataset at ${RAW_DATA_DIR}."
    exit 1
fi
echo "Preprocessing parquet file: ${PARQUET_FILE_PATH}"

python preprocess/process_parquet.py --file_path "${PARQUET_FILE_PATH}" --output_dir "${PROCESSED_DATA_DIR}"
echo "--- Preprocessing complete. ---"
