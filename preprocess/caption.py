
import os
import argparse
from natsort import natsorted
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm

def caption_images_in_directory(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
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

    image_files = natsorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    output_file = os.path.join(output_dir, "caption.csv")
    with open(output_file, 'a') as f:
        if os.path.getsize(output_file) == 0:
            f.write("filename,caption\n")
        
        print(f"Generating captions for {len(image_files)} images...")
        for filename in tqdm(image_files, desc="Captioning Images"):
            image_path = os.path.join(image_dir, filename)
            try:
                raw_image = Image.open(image_path).convert('RGB')
                
                # Generate caption
                inputs = processor(raw_image, return_tensors="pt").to(device)
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                caption = ','.join(caption).rstrip()
                # Write to file
                f.write(f"{image_path},'{caption.rstrip()}'\n")

            except Exception as e:
                print(f"Could not process {filename}: {e}")

    print(f"\nCaptions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for the image dataset.")
    parser.add_argument("--image_dir", type=str, help="The directory to the image dataset.")
    parser.add_argument("--output_dir", type=str, help="The path to the output file to save the captions (e.g., data/).")
    
    args = parser.parse_args()
    caption_images_in_directory(args.image_dir, args.output_dir)
