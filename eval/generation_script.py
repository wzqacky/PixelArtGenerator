import os
import torch
import argparse
from pathlib import Path
from diffusers import DiffusionPipeline, UNet2DConditionModel
import json
from typing import List

# ===================== Configuration Parameters =====================
class Config:
    # Default Model Paths
    BASE_MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
    FULL_FINETUNED_MODEL_PATH = "checkpoints/pixel-art-model_sdxl-filtered-dataset"
    LORA_MODEL_PATH = "checkpoints/pixel-art-model_sdxl_lora-filterted-dataset"
    
    # Generation Output Folders
    BASE_MODEL_OUTPUT_FOLDER = "./base_model_images"
    FULL_FINETUNED_OUTPUT_FOLDER = "./full_finetuned_images"
    LORA_MODEL_OUTPUT_FOLDER = "./lora_model_images"
    
    # Pixel Art Prompts - all include pixel art requirements
    PIXEL_ART_PROMPTS: List[str] = [
        "8-bit pixel art of a fantasy castle, retro game style, vibrant colors, clean pixels, no anti-aliasing",
        "pixel art landscape with mountains and rivers, 16-bit, top-down view, minimal color palette",
        "retro pixel art character: warrior with sword, 8-bit, side profile, simple background",
        "pixel art city skyline at night, neon lights, 16-bit, pixel-perfect, no blur",
        "8-bit pixel art animal: fox in a forest, cute style, blocky pixels, limited color count",
        "pixel art space scene with planets and stars, retro arcade style, 16-bit, high contrast",
        "pixel art food: burger and fries, 8-bit, flat design, vibrant colors, clean edges",
        "retro pixel art dungeon interior, 16-bit, top-down, torch light, minimal details",
        "pixel art mermaid in ocean, 8-bit, colorful scales, simple background, no gradients",
        "pixel art car race, retro game style, 16-bit, pixel-perfect, dynamic pose"
    ]
    
    # Generation Hyperparameters
    NUM_IMAGES_PER_PROMPT = 3
    IMAGE_RESOLUTION = (512, 512)
    INFERENCE_STEPS = 30
    GUIDANCE_SCALE = 7.5
    SEED_OFFSET = 42
    BATCH_SIZE = 1
    NEGATIVE_PROMPT = "3d, smooth, blurry, soft, intricate, artifacts, low-quality, "

# ===================== Utility Functions =====================
def create_output_folders(output_path: str) -> None:
    """Create output folder if it doesn't exist"""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"Output folder ready: {output_path}")

def load_model(model_type: str, base_model: str, model_path: str = None, use_lora: bool = False) -> DiffusionPipeline:
    """Load appropriate model based on type"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_type} model to {device}")
    
    try:
        if model_type == "base":
            pipe = DiffusionPipeline.from_pretrained(base_model)
            
        elif model_type == "full_finetuned":
            if not model_path:
                raise ValueError("Model path is required for full finetuned model")
            pipe = DiffusionPipeline.from_pretrained(model_path)
            
        elif model_type == "lora":
            if not model_path or not base_model:
                raise ValueError("Both model path and base model are required for LoRA model")
            pipe = DiffusionPipeline.from_pretrained(base_model)
            pipe.load_lora_weights(model_path)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Optimization
        pipe.to(device)
        if device == "cuda":
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing()
            
        print(f"Successfully loaded {model_type} model")
        return pipe
        
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        raise SystemExit(1)

def generate_model_images(
    model_type: str, 
    pipeline: DiffusionPipeline, 
    prompts: List[str], 
    output_folder: str, 
    num_images_per_prompt: int,
    seed_offset: int
) -> None:
    """Generate images for a specific model type"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_images = len(prompts) * num_images_per_prompt
    print(f"\nStarting {model_type} model generation (total images: {total_images})")
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nGenerating for prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
        
        # Generate multiple images per prompt with different seeds
        seeds = [seed_offset + (prompt_idx * num_images_per_prompt) + i for i in range(num_images_per_prompt)]
        
        for seed_idx, seed in enumerate(seeds):
            # Set seed for reproducibility
            generator = torch.Generator(device).manual_seed(seed)
            
            # Generate image
            with torch.no_grad():
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=Config.NEGATIVE_PROMPT,
                    height=Config.IMAGE_RESOLUTION[1],
                    width=Config.IMAGE_RESOLUTION[0],
                    num_inference_steps=Config.INFERENCE_STEPS,
                    guidance_scale=Config.GUIDANCE_SCALE,
                    generator=generator,
                    num_images_per_prompt=1
                ).images[0]
            
            # Save image
            image_filename = f"{model_type}_prompt_{prompt_idx}_seed_{seed}.png"
            image_path = os.path.join(output_folder, image_filename)
            image.save(image_path)
            
            print(f"Generated: {image_path}")

# ===================== Main Workflow =====================
def main():
    parser = argparse.ArgumentParser(description="Generate pixel art images using different model types.")
    parser.add_argument("--models", type=str, nargs='+', 
                      choices=["base", "full_finetuned", "lora"],
                      default=["base", "full_finetuned", "lora"],
                      help="Specify which models to use for generation")
    parser.add_argument("--num_images", type=int, 
                      default=Config.NUM_IMAGES_PER_PROMPT,
                      help="Number of images to generate per prompt")
    parser.add_argument("--seed_offset", type=int, 
                      default=Config.SEED_OFFSET,
                      help="Base seed for reproducibility")
    
    args = parser.parse_args()

    # Create output folders
    if "base" in args.models:
        create_output_folders(Config.BASE_MODEL_OUTPUT_FOLDER)
    if "full_finetuned" in args.models:
        create_output_folders(Config.FULL_FINETUNED_OUTPUT_FOLDER)
    if "lora" in args.models:
        create_output_folders(Config.LORA_MODEL_OUTPUT_FOLDER)

    # Generate images for each selected model type
    if "base" in args.models:
        base_pipeline = load_model("base", Config.BASE_MODEL_PATH)
        generate_model_images(
            "base", 
            base_pipeline, 
            Config.PIXEL_ART_PROMPTS,
            Config.BASE_MODEL_OUTPUT_FOLDER,
            args.num_images,
            args.seed_offset
        )

    if "full_finetuned" in args.models:
        finetuned_pipeline = load_model(
            "full_finetuned", 
            Config.BASE_MODEL_PATH,
            Config.FULL_FINETUNED_MODEL_PATH
        )
        generate_model_images(
            "full_finetuned", 
            finetuned_pipeline, 
            Config.PIXEL_ART_PROMPTS,
            Config.FULL_FINETUNED_OUTPUT_FOLDER,
            args.num_images,
            args.seed_offset + 1000  # Unique seed offset to avoid overlap
        )

    if "lora" in args.models:
        lora_pipeline = load_model(
            "lora", 
            Config.BASE_MODEL_PATH,
            Config.LORA_MODEL_PATH,
            use_lora=True
        )
        generate_model_images(
            "lora", 
            lora_pipeline, 
            Config.PIXEL_ART_PROMPTS,
            Config.LORA_MODEL_OUTPUT_FOLDER,
            args.num_images,
            args.seed_offset + 2000  # Unique seed offset to avoid overlap
        )

    print("\n=== Generation Complete ===")
    if "base" in args.models:
        print(f"Base model images saved to: {Config.BASE_MODEL_OUTPUT_FOLDER}")
    if "full_finetuned" in args.models:
        print(f"Full finetuned model images saved to: {Config.FULL_FINETUNED_OUTPUT_FOLDER}")
    if "lora" in args.models:
        print(f"LoRA model images saved to: {Config.LORA_MODEL_OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()