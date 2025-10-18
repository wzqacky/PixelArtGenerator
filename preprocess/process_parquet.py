
import pandas as pd
import argparse
import os
import requests
from tqdm import tqdm

def download_image(url, folder, filename):
    """
    Downloads an image from a URL and saves it to a folder.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filepath = os.path.join(folder, filename)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")


def read_parquet_and_download_images(file_path, output_dir):
    """
    Reads a Parquet file and downloads images from the 'image_url' column.
    """
    try:
        print(f"Reading data from: {file_path}")
        df = pd.read_parquet(file_path)
        print("Successfully loaded data.")
        
        if 'image_url' not in df.columns:
            print("Error: 'image_url' column not found in the Parquet file.")
            return

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
            image_url = row['image_url']
            caption = row['title']
            if pd.notna(image_url):
                # Creating a unique filename for each image
                filename = f"{caption}.jpg"
                download_image(image_url, output_dir, filename)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a Parquet file and download images.")
    parser.add_argument("--file_path", type=str, help="The path to the Parquet file.")
    parser.add_argument("--output_dir", type=str, help="The directory to save the downloaded images.")
    
    args = parser.parse_args()
    read_parquet_and_download_images(args.file_path, args.output_dir)
