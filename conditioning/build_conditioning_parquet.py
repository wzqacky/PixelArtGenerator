import argparse
import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd

from utils import build_canny_map, build_color_palette

def generate(dataset, args):
    processed_rows = []
    for idx, item in enumerate(tqdm(dataset)):
        if args.target_condition == "canny":
            conditioning_bytes = build_canny_map(item['path'], src_byte=item['image'], plot=args.plot)
        elif args.target_condition == "palette":
            conditioning_bytes = build_color_palette(item['path'], src_byte=item['image'], plot=args.plot)
        if not conditioning_bytes:
            print(f"Error processing the conditioning images for {item['path']}, skipping.")
            continue
        new_row = item.copy()
        new_row['conditioning_image'] = conditioning_bytes
        processed_rows.append(new_row)

    if not args.output_dataset_path:
        output_dataset_path = Path(args.dataset_path).parent / f"{args.target_condition}_{Path(args.dataset_path).name}"
    else:
        output_dataset_path = args.output_dataset_path
    if processed_rows:
        df = pd.DataFrame(processed_rows)
        df.to_parquet(output_dataset_path)
        print(f"Saved new dataset with {args.target_condition} as conditioning images to {output_dataset_path}")
    else:
        print("No images were processed successfully. Output file not created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset parquet file")
    parser.add_argument("--output_dataset_path", type=str, help="Path to the output dataset parquet file")
    parser.add_argument("--target_condition", type=str, required=True, help="Target conditoning image type to generate.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the conditioning image results.")
    args = parser.parse_args()
    if args.target_condition.lower() not in ["canny", "palette"]:
        raise ValueError("Please select 'canny' or 'palette' to be the conditioning images.")
    if args.plot:
        visualization_path = f"visualization/{args.target_condition}"
        os.makedirs(visualization_path, exist_ok=True)
    dataset = load_dataset("parquet", data_files=args.dataset_path)['train']
    generate(dataset, args)
