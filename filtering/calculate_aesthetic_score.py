import argparse
import os
from os.path import expanduser
import io
import csv
from urllib.request import urlretrieve
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

def main(args):
    # Load CLIP model and processor
    clip_model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)

    # Load the aesthetic predictor model
    aesthetic_predictor = get_aesthetic_model(clip_model="vit_b_32")
    aesthetic_predictor.to(device)
    aesthetic_predictor.eval()

    # Load the dataset
    dataset = load_dataset("parquet", data_files=args.dataset_path)

    # Process images and calculate aesthetic score
    scores = []
    for i, item in enumerate(tqdm(dataset['train'])):
        image_bytes = item['image']
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Get CLIP image features
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        score = aesthetic_predictor(image_features)
        scores.append((item['path'], score.item()))

    # Sort by score and save
    scores.sort(key=lambda x: x[1], reverse=True)
    with open(args.output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "score"])
        writer.writerows(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate aesthetic score for images.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset parquet file.")
    parser.add_argument("--output_file", type=str, default="data/aesthetic_scores.csv", help="Output file to save results.")
    args = parser.parse_args()
    main(args)
