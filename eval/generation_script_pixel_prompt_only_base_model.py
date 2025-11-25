import os
import torch
import argparse
from pathlib import Path
from diffusers import DiffusionPipeline
from typing import List

# ===================== Configuration Parameters =====================
class Config:
    # Model Paths (matches your inference script)
    BASE_MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
    FULL_FINETUNED_MODEL_PATH = "wzqacky/pixel-art-model-sdxl"
    LORA_MODEL_PATH = "wzqacky/pixel-art-model-sdxl-lora"
    
    # Output Folders (separate for each model type)
    BASE_MODEL_OUTPUT_FOLDER = "./base_model_images"
    FULL_FINETUNED_OUTPUT_FOLDER = "./full_finetuned_images"
    LORA_MODEL_OUTPUT_FOLDER = "./lora_model_images"
    
    # Prompt Lists (differentiated by model type)
    # Base model: Pixel art-specific prompts (needs guidance for pixel style)
    BASE_MODEL_PROMPTS: List[str] = [
        "8-bit fantasy castle, retro game style, vibrant colors, clean pixels, no anti-aliasing",
        "16-bit landscape with mountains and rivers, top-down view, minimal color palette",
        "8-bit warrior with sword, side profile, simple background, blocky pixels",
        "16-bit city skyline at night, neon lights, pixel-perfect, no blur",
        "8-bit fox in a forest, cute style, limited color count, sharp edges",
        "16-bit space scene with planets and stars, retro arcade style, high contrast",
        "8-bit burger and fries, flat design, vibrant colors, clean pixels",
        "16-bit dungeon interior, top-down, torch light, minimal details",
        "8-bit mermaid in ocean, colorful scales, simple background, no gradients",
        "16-bit car race, retro game style, dynamic pose, pixel-perfect"
    ]
    
    # Full finetuned/LoRA models: General prompts (no pixel keywords—models are specialized)
    SPECIALIZED_MODEL_PROMPTS: List[str] = [
        "fantasy castle, retro game style, vibrant colors, simple background",
        "landscape with mountains and rivers, top-down view, minimal color palette",
        "warrior with sword, side profile, simple background, bold design",
        "city skyline at night, neon lights, sharp edges, no blur",
        "fox in a forest, cute style, vibrant colors, clean design",
        "space scene with planets and stars, retro arcade style, high contrast",
        "burger and fries, flat design, vibrant colors, simple composition",
        "dungeon interior, top-down, torch light, minimal details",
        "mermaid in ocean, colorful scales, simple background",
        "car race, retro game style, dynamic pose, bold lines"
    ]
    
    # Generation Hyperparameters (consistent across models for fair comparison)
    NUM_IMAGES_PER_PROMPT = 3  # Images generated per prompt
    IMAGE_RESOLUTION = (512, 512)  # Square resolution (ideal for pixel art)
    INFERENCE_STEPS = 30  # Balance of speed and quality
    GUIDANCE_SCALE = 7.5  # Balances prompt adherence and creativity
    NEGATIVE_PROMPT = "3d, smooth, blurry, soft, intricate, artifacts, low-quality, gradients"  # Avoid non-pixel features
    SEED_OFFSET = 42  # Base seed for reproducibility (unique offsets per model)

# ===================== Utility Functions =====================
def create_output_folders(output_path: str) -> None:
    """Create output folder if it doesn't exist (no overwite warning—safe for new generation)"""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"Output folder initialized: {output_path}")

def load_model(model_type: str) -> DiffusionPipeline:
    """Load model based on type (base/full_finetuned/LoRA) with matching logic from inference.py"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {model_type} model to {device}...")
    
    try:
        if model_type == "base":
            # Load base model directly (no fine-tuning)
            pipe = DiffusionPipeline.from_pretrained(Config.BASE_MODEL_PATH)
            
        elif model_type == "full_finetuned":
            # Load full fine-tuned model (specialized for pixel art)
            pipe = DiffusionPipeline.from_pretrained(Config.FULL_FINETUNED_MODEL_PATH)
            
        elif model_type == "lora":
            # Load LoRA + base model (LoRA adds pixel art capability)
            pipe = DiffusionPipeline.from_pretrained(Config.BASE_MODEL_PATH)
            pipe.load_lora_weights(Config.LORA_MODEL_PATH)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Optimize for speed/memory (GPU-only)
        pipe.to(device)
        if device == "cuda":
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing()  # Reduce VRAM usage
            
        print(f"Successfully loaded {model_type} model")
        return pipe
        
    except Exception as e:
        print(f"Error loading {model_type} model: {str(e)}")
        raise SystemExit(1)

def generate_model_images(model_type: str, pipeline: DiffusionPipeline) -> None:
    """Generate images for a model type with matching prompts and settings"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Select correct prompts based on model type
    if model_type == "base":
        prompts = Config.BASE_MODEL_PROMPTS
        output_folder = Config.BASE_MODEL_OUTPUT_FOLDER
        seed_offset = Config.SEED_OFFSET  # Unique seed offset to avoid overlap
    elif model_type == "full_finetuned":
        prompts = Config.SPECIALIZED_MODEL_PROMPTS
        output_folder = Config.FULL_FINETUNED_OUTPUT_FOLDER
        seed_offset = Config.SEED_OFFSET + 1000  # Avoid seed overlap with base model
    elif model_type == "lora":
        prompts = Config.SPECIALIZED_MODEL_PROMPTS
        output_folder = Config.LORA_MODEL_OUTPUT_FOLDER
        seed_offset = Config.SEED_OFFSET + 2000  # Avoid seed overlap with other models
    
    total_images = len(prompts) * Config.NUM_IMAGES_PER_PROMPT
    print(f"\nStarting {model_type} model generation (total images: {total_images})")
    print(f"Using prompts: {prompts[:2]}... (total {len(prompts)} prompts)")
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:60]}...")
        
        # Generate multiple images per prompt (different seeds for variety)
        seeds = [seed_offset + (prompt_idx * Config.NUM_IMAGES_PER_PROMPT) + i for i in range(Config.NUM_IMAGES_PER_PROMPT)]
        
        for seed_idx, seed in enumerate(seeds):
            # Set fixed seed for reproducibility
            generator = torch.Generator(device).manual_seed(seed)
            
            # Generate image with consistent hyperparameters
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=Config.NEGATIVE_PROMPT,
                    height=Config.IMAGE_RESOLUTION[1],
                    width=Config.IMAGE_RESOLUTION[0],
                    num_inference_steps=Config.INFERENCE_STEPS,
                    guidance_scale=Config.GUIDANCE_SCALE,
                    generator=generator,
                    num_images_per_prompt=1
                )
                image = result.images[0]
            
            # Save image with descriptive filename
            image_filename = f"{model_type}_prompt_{prompt_idx:02d}_seed_{seed}.png"
            image_path = os.path.join(output_folder, image_filename)
            image.save(image_path)
            
            print(f"Generated: {image_path} (seed: {seed})")

# ===================== Main Workflow =====================
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate pixel art test data for base/full-finetuned/LoRA models")
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+", 
        choices=["base", "full_finetuned", "lora"],
        default=["base", "full_finetuned", "lora"],
        help="Specify which models to use (e.g., --models base lora)"
    )
    parser.add_argument(
        "--num_images_per_prompt", 
        type=int, 
        default=Config.NUM_IMAGES_PER_PROMPT,
        help="Number of images to generate per prompt (default: 3)"
    )
    parser.add_argument(
        "--seed_offset", 
        type=int, 
        default=Config.SEED_OFFSET,
        help="Base seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    Config.NUM_IMAGES_PER_PROMPT = args.num_images_per_prompt
    Config.SEED_OFFSET = args.seed_offset
    
    # Create output folders for selected models
    for model_type in args.models:
        if model_type == "base":
            create_output_folders(Config.BASE_MODEL_OUTPUT_FOLDER)
        elif model_type == "full_finetuned":
            create_output_folders(Config.FULL_FINETUNED_OUTPUT_FOLDER)
        elif model_type == "lora":
            create_output_folders(Config.LORA_MODEL_OUTPUT_FOLDER)
    
    # Generate images for each selected model
    for model_type in args.models:
        pipeline = load_model(model_type)
        generate_model_images(model_type, pipeline)
    
    # Final summary
    print("\n=== Generation Complete ===")
    for model_type in args.models:
        if model_type == "base":
            print(f"Base model images: {Config.BASE_MODEL_OUTPUT_FOLDER}")
        elif model_type == "full_finetuned":
            print(f"Full fine-tuned model images: {Config.FULL_FINETUNED_OUTPUT_FOLDER}")
        elif model_type == "lora":
            print(f"LoRA model images: {Config.LORA_MODEL_OUTPUT_FOLDER}")
    print(f"\nTotal images per model: {len(Config.BASE_MODEL_PROMPTS) * Config.NUM_IMAGES_PER_PROMPT}")
    print("Use the evaluation script to compare pixelation and aesthetic scores!")

if __name__ == "__main__":
    main()