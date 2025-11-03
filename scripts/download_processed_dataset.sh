#!/bin/bash
set -e

REPO_ID="wzqacky/captioned_pixelart_images"
LOCAL_DIR="data"
FILE_NAME="captioned_pixelart_images.parquet"

echo ">>> Downloading processed dataset file: ${FILE_NAME} from ${REPO_ID}..."

python preprocess/download_hf.py \
    --repo_id "${REPO_ID}" \
    --local_dir "${LOCAL_DIR}" \
    --repo_type dataset \
    --file_name "${FILE_NAME}"

echo "--- Download complete. ---"
