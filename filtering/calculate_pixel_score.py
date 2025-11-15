import argparse
import torch
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from PIL import Image
import io
from tqdm import tqdm
import csv

def main(args):
    # Load CLIP model and processor
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load the dataset
    dataset = load_dataset("parquet", data_files=args.dataset_path)

    # Prepare the text input
    text_inputs = processor(text=["a pixel-art image"], return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Process images and calculate similarity
    similarities = []
    for item in tqdm(dataset['train']):
        image_bytes = item['image']
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        similarity = torch.cosine_similarity(text_features, image_features)
        similarities.append((item['path'], similarity.item()))

    # Sort by similarity and save
    similarities.sort(key=lambda x: x[1], reverse=True)

    with open(args.output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "score"])
        writer.writerows(similarities)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find pixel art images.")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model to use.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset parquet file.")
    parser.add_argument("--output_file", type=str, default="data/pixel_art_similarity.csv", help="Output file to save results.")
    args = parser.parse_args()
    main(args)
