#!/bin/bash
set -e

upload_to_hf=false
# Parse command-line arguments
for arg in "$@"
do
    case $arg in
        --upload_to_hf)
        upload_to_hf=true
        shift # Remove --upload_to_hf from processing
        ;;
    esac
done

PROCESSED_DATA_DIR_KAGGLE="data/pixel-art-512x512"
PROCESSED_DATA_DIR_HF="data/pixilart_processed"
CAPTIONED_DATA_PATH="data/captioned_pixelart.parquet"
REPO_ID="wzqacky/captioned_pixelart_images"

echo ">>> Generating captions for images..."
python preprocess/caption_and_build_parquet.py --image_dirs "${PROCESSED_DATA_DIR_KAGGLE}" "${PROCESSED_DATA_DIR_HF}" --output_path "${CAPTIONED_DATA_PATH}"
echo "--- Captioning complete. ---"

if [ "$upload_to_hf" = true ]; then
    echo ">>> Uploading dataset to HuggingFace Hub..."
    if [ ! -f "${CAPTIONED_DATA_PATH}" ]; then
        echo "Error: Captioned data file not found at ${CAPTIONED_DATA_PATH}. Please ensure your run on captioning is complete."
        exit 1
    fi
    python preprocess/upload_to_hf.py --dataset_path "${CAPTIONED_DATA_PATH}" --repo_id "${REPO_ID}"
    echo "--- Upload complete. ---"
fi
