
import os
import argparse
from natsort import natsorted
import re
import io
from PIL import Image
import pandas as pd
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm

def filter_caption(caption):
    words_to_remove = [r'\bpixel art\b', r'\bpixelated\b', r'\bpixel-art\b', r'\bpixel\b']
    for word_pattern in words_to_remove:
        # Special handling for the spaces before and after the word
        caption = re.sub(r'\s*' + word_pattern + r'\s*', ' ', caption, flags=re.IGNORECASE)
    return caption.strip()

def caption_images_in_directory(image_dirs, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    """
    Generates captions for all images in a directory and saves them to a file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the processor and model
    print("Loading image captioning model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b").to(device)
    print("Model loaded.")

    image_files = []
    for image_dir in image_dirs:
        for f in os.listdir(image_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(image_dir, f))

    image_files = natsorted(image_files)
    captions_data = []
    print(f"Generating captions for {len(image_files)} images...")
    for image_path in tqdm(image_files, desc="Captioning Images"):
        try:
            raw_image = Image.open(image_path).convert('RGB')
            
            # Generate caption
            inputs = processor(raw_image, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            caption = caption.replace(",", "").rstrip() # remove comma, and newline character at the end
            print(f"Caption is {caption}")
            filtered_caption = filter_caption(caption)

            buffer = io.BytesIO()
            raw_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            captions_data.append({"image": image_bytes, 'caption': filtered_caption})

        except Exception as e:
            print(f"Could not process {image_path}: {e}")
        
    if captions_data:
        df = pd.DataFrame(captions_data)
        df.to_parquet(output_path, index=False)
        print(f"\nCaptioned images parquet saved to {output_path}")
    else:
        print("No captions were generated, failed to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for the image dataset.")
    parser.add_argument("--image_dirs", type=str, nargs='+', help="The directories to the image dataset.")
    parser.add_argument("--output_path", type=str, help="The path to the output file to save the captions (e.g., data/captioned_pixelart_images.parquet).")
    
    args = parser.parse_args()
    caption_images_in_directory(args.image_dirs, args.output_path)
