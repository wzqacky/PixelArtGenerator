import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def main(args):
    input_path = args.input_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    dist_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_distribution.png")
    img_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_images.png")

    # Read the data from the input file
    try:
        df = pd.read_csv(input_path, sep=",", skiprows=[0], header=None, names=["path", "score"])
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # Calculate statistics
    mean_score = df["score"].mean()
    q1 = df["score"].quantile(0.25)
    q2 = df["score"].quantile(0.5)
    q3 = df["score"].quantile(0.75)

    # Score distribution plot
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    ax1.hist(df["score"], bins=50, density=True, alpha=0.6, color='g')
    ax1.axvline(mean_score, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
    ax1.axvline(q1, color='b', linestyle=':', linewidth=2, label=f'Q1: {q1:.2f}')
    ax1.axvline(q2, color='purple', linestyle=':', linewidth=2, label=f'Q2 (Median): {q2:.2f}')
    ax1.axvline(q3, color='orange', linestyle=':', linewidth=2, label=f'Q3: {q3:.2f}')
    ax1.set_title('Image Score Distribution')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Density')
    ax1.legend()
    plt.tight_layout()

    plt.savefig(dist_output_path)
    print(f"Distribution plot saved to {dist_output_path}")
    plt.close(fig1)


    # Example Images
    example_images = {}
    try:
        example_images["Below Q1"] = df[df["score"] < q1].sample(1).iloc[0]
    except ValueError:
        print("Warning: No images found below Q1.")
    try:
        example_images["Between Q1 and Q2"] = df[(df["score"] >= q1) & (df["score"] < q2)].sample(1).iloc[0]
    except ValueError:
        print("Warning: No images found between Q1 and Q2.")
    try:
        example_images["Between Q2 and Q3"] = df[(df["score"] >= q2) & (df["score"] < q3)].sample(1).iloc[0]
    except ValueError:
        print("Warning: No images found between Q2 and Q3.")
    try:
        example_images["Above Q3"] = df[df["score"] >= q3].sample(1).iloc[0]
    except ValueError:
        print("Warning: No images found above Q3.")

    if not example_images:
        print("No example images to plot.")
        return

    num_images = len(example_images)
    ncols = min(num_images, 2)
    nrows = (num_images + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    for ax, (label, data) in zip(axes, example_images.items()):
        img = mpimg.imread(data["path"])
        ax.imshow(img)
        ax.set_title(f"{label}\nScore: {data['score']:.2f}")
        ax.axis('off')

    fig2.suptitle('Example Images from Score Tiers', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(img_output_path)
    print(f"Example images plot saved to {img_output_path}")
    plt.close(fig2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image scores using Matplotlib.")
    parser.add_argument("--input_path", type=str, help="Path to the input .csv file with image paths and scores.")
    parser.add_argument("--output_dir", type=str, default="visualization/", help="Directory for the output plot image files.")
    args = parser.parse_args()
    main(args)