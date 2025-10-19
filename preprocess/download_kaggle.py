import os
import argparse
import kagglehub
import tempfile
import shutil

def download_from_kaggle(dataset_id, local_dir):
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        # Set the cache directory for kagglehub to our temporary directory
        os.environ['KAGGLEHUB_CACHE'] = temp_cache_dir

        # Download latest version
        download_path = kagglehub.dataset_download(dataset_id)

        os.makedirs(local_dir, exist_ok=True)
        # Transfer all the files to the designated directory, and remove the temp directory
        for item in os.listdir(download_path):
            source_item_path = os.path.join(download_path, item)
            destination_item_path = os.path.join(local_dir, item)
            shutil.move(source_item_path, destination_item_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset from Kaggle and move it to a specific folder.")
    parser.add_argument("--dataset_id", type=str, help="The dataset ID on Kaggle (e.g., 'user/dataset-name').")
    parser.add_argument("--local_dir", type=str, help="The local directory to save the dataset files.")
    args = parser.parse_args()

    download_from_kaggle(args.dataset_id, args.local_dir)
