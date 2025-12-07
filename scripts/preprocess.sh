#!/bin/bash
set -e


PROCESSED_DATA_DIR_KAGGLE="data/pixel-art-512x512"
PROCESSED_DATA_DIR_HF="data/pixilart_processed"
CAPTIONED_DATA_PATH="data/captioned_pixelart.parquet"

echo ">>> Generating captions for images..."
python preprocess/caption_and_build_parquet.py --image_dirs "${PROCESSED_DATA_DIR_KAGGLE}" "${PROCESSED_DATA_DIR_HF}" --output_path "${CAPTIONED_DATA_PATH}"
echo "--- Captioning complete. ---"
