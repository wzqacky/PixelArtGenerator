
import pandas as pd
import argparse
import re

def filter_captions(input_csv_path, output_csv_path):
    """
    Reads a CSV file, removes "pixel art" related wordings from the 'caption' column,
    and saves the modified data to a new CSV file.
    """
    try:
        df = pd.read_csv(input_csv_path)

        if 'caption' not in df.columns:
            print("Error: 'caption' column not found in the input CSV file.")
            return

        # Define words to remove (case-insensitive)
        words_to_remove = [r'\bpixel art\b', r'\bpixelated\b', r'\bpixel-art\b', r'\bpixel\b']

        # Function to remove words from a caption
        def clean_caption(caption):
            for word_pattern in words_to_remove:
                caption = re.sub(r'\s*' + word_pattern + r'\s*', ' ', caption, flags=re.IGNORECASE)
            return caption.strip()
        df['caption'] = df['caption'].apply(clean_caption)

        df.to_csv(output_csv_path, index=False)
        print(f"Filtered captions saved to {output_csv_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter captions in a CSV file.")
    parser.add_argument("--input_csv", type=str, help="The path to the input caption CSV file.")
    parser.add_argument("--output_csv", type=str, help="The path to the output CSV file for filtered captions.")

    args = parser.parse_args()
    filter_captions(args.input_csv, args.output_csv)
