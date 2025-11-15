import pandas as pd
import argparse
from pathlib import Path

def main(args):
    input_parquet_path = Path(args.input_parquet_path)
    csv_path = Path(args.csv_file)
    output_parquet_path = Path(args.output_parquet_path)

    # Load the datasets
    print(f"Loading Parquet file from: {input_parquet_path}")
    try:
        main_df = pd.read_parquet(input_parquet_path)
    except FileNotFoundError:
        print(f"Error: Parquet file not found at {input_parquet_path}")
        return
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        return

    print(f"Loading CSV file from: {csv_path}")
    try:
        scores_df = pd.read_csv(csv_path, sep=",", skiprows=[0], header=None, names=["path", "score"])
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    threshold= scores_df["score"].quantile(0.25)

    # Filtering based on scores
    print(f"Filtering scores dataframe for entries with score >= {threshold}...")
    filtered_scores_df = scores_df[scores_df["score"] >= threshold]
    
    num_above_threshold = len(filtered_scores_df)
    print(f"Filtered dataset contains {num_above_threshold} entries.")

    # Merging
    filtered_main_df = main_df.merge(
        filtered_scores_df['path'],
        left_on='path',
        right_on='path',
        how="inner",
    )
    # Save the filtered df
    try:
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_main_df.to_parquet(output_parquet_path, index=False)
        print(f"Filtered dataset saved in {output_parquet_path}.")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter a Parquet file based on aesthetic scores from a CSV file."
    )
    parser.add_argument("--input_parquet_path", type=str, required=True, help="Path to the input Parquet file.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file containing scores.")
    parser.add_argument("--output_parquet_path", type=str, required=True, help="Path to save the filtered output Parquet file.")
    args = parser.parse_args()
    main(args)
    
