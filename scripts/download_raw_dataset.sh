#!/bin/bash
set -e

RAW_DATA_DIR_HF="data/pixilart"
PROCESSED_DATA_DIR_KAGGLE="data/pixel-art-512x512"
PROCESSED_DATA_DIR_HF="data/pixilart_processed"

echo ">>> Downloading raw dataset from HuggingFace and KaggleHub..."
if [ -d "${PROCESSED_DATA_DIR_KAGGLE}" ]; then
    echo "Kaggle dataset already exists."
else
    python kaggle/download_kaggle.py --dataset_id artvandaley/curated-pixel-art-512x512 --local_dir ${PROCESSED_DATA_DIR_KAGGLE}
fi

if [ -d "${RAW_DATA_DIR_HF}" ]; then
    echo "HuggingFace dataset already exists."
else
    python hf/download_hf.py --repo_id bghira/free-to-use-pixelart --local_dir ${RAW_DATA_DIR_HF} --repo_type dataset
fi

PARQUET_FILE_PATH=$(find "${RAW_DATA_DIR_HF}" -type f -name "*.parquet" | head -n 1)
if [ -z "${PARQUET_FILE_PATH}" ]; then
    echo "Error: No .parquet file found in the downloaded dataset at ${RAW_DATA_DIR_HF}."
    exit 1
fi
echo "Preprocessing parquet file: ${PARQUET_FILE_PATH}"

python preprocess/process_parquet.py --file_path "${PARQUET_FILE_PATH}" --output_dir "${PROCESSED_DATA_DIR_HF}"
echo "--- Downloading complete. ---"