import pandas as pd
import argparse
from pathlib import Path

def main(args):
    input_parquet_path = Path(args.input_parquet_path)
    csv_files = [Path(p) for p in args.csv_files]
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

    if not csv_files:
        print("Error: No CSV files provided.")
        return

    # Load and merge score files
    print("Loading and merging score files...")
    merged_scores_df = None
    for csv_path in csv_files:
        print(f"Processing {csv_path}...")
        try:
            score_name = csv_path.stem
            current_scores_df = pd.read_csv(csv_path, sep=",", skiprows=[0], header=None, names=["path", score_name])
            if merged_scores_df is None:
                merged_scores_df = current_scores_df
            else:
                merged_scores_df = pd.merge(merged_scores_df, current_scores_df, on="path", how="inner")
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            return
        except Exception as e:
            print(f"Error loading or merging CSV file {csv_path}: {e}")
            return
    
    if merged_scores_df is None or merged_scores_df.empty:
        print("No common entries found across all score files. Output will be empty.")
        filtered_main_df = pd.DataFrame(columns=main_df.columns)
    else:
        print(f"Found {len(merged_scores_df)} entries with all specified scores.")
        print("Filtering based on 25th percentiles...")

        combined_filter = pd.Series([True] * len(merged_scores_df), index=merged_scores_df.index)

        score_columns = [col for col in merged_scores_df.columns if col != 'path']

        for score_col in score_columns:
            threshold = merged_scores_df[score_col].quantile(0.25)
            combined_filter &= (merged_scores_df[score_col] >= threshold)

        filtered_scores_df = merged_scores_df[combined_filter]
        
        num_above_threshold = len(filtered_scores_df)
        print(f"Filtered dataset contains {num_above_threshold} entries passing all thresholds.")

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
        description="Filter a Parquet file based on scores from multiple CSV files."
    )
    parser.add_argument("--input_parquet_path", type=str, required=True, help="Path to the input Parquet file.")
    parser.add_argument("--csv_files", type=str, nargs='+', required=True, help="Paths to the CSV files containing scores.")
    parser.add_argument("--output_parquet_path", type=str, required=True, help="Path to save the filtered output Parquet file.")
    args = parser.parse_args()
    main(args)
