import os
import sys
from typing import Dict, Any, List
import argparse
import yaml
import torch
import csv
from diffusers import DiffusionPipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_output_folders, load_model, load_prompts_from_csv


def generate_model_images(model_type: str, pipeline: DiffusionPipeline, config: Dict[str, Any]) -> None:
    """Generate images for a model type with matching prompts and settings from config"""
    seed_offset = config["seed_offset"]
    
    num_images_per_prompt = config["num_images_per_prompt"]
    if model_type == "base":
        prompts = config["base_model_prompts"]
        output_folder = config[f"{model_type}_output_folder"]
    elif model_type == "full_finetuned":
        prompts = config["specialized_model_prompts"]
        output_folder = config[f"{model_type}_output_folder"]
        seed_offset += 1000
    elif model_type == "lora":
        prompts = config["specialized_model_prompts"]
        output_folder = config[f"{model_type}_output_folder"]
        seed_offset += 2000

    total_images = len(prompts) * num_images_per_prompt
    print(f"\nStarting {model_type} model generation (total images: {total_images})")
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:60]}...")
        seeds = [seed_offset + (prompt_idx * num_images_per_prompt) + i for i in range(num_images_per_prompt)]

        for seed_idx, seed in enumerate(seeds):
            generator = torch.Generator(pipeline.device).manual_seed(seed)
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=config["negative_prompt"],
                    height=config["image_resolution"][1],
                    width=config["image_resolution"][0],
                    num_inference_steps=config["inference_steps"],
                    guidance_scale=config["guidance_scale"],
                    generator=generator,
                    num_images_per_prompt=1
                )
                image = result.images[0]
            image_filename = f"{model_type}_condition_{prompt_idx:02d}_seed_{seed}.png"
            image_path = os.path.join(output_folder, image_filename)
            image.save(image_path)
            print(f"Generated: {image_path} (seed: {seed})")

def main():
    parser = argparse.ArgumentParser(description="Generate pixel art test data for various models.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="eval/generation_config.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--model_types", 
        type=str, 
        nargs="+", 
        choices=["base", "full_finetuned", "lora"],
        default=None,
        help="Specify which models to use. If not specified, all are used."
    )

    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        raise sys.exit(1)

    prompts = load_prompts_from_csv(config["prompts_csv_path"])
    config["base_model_prompts"] = prompts["base"]
    config["specialized_model_prompts"] = prompts["specialized"]
    
    if not args.model_types:
        print("No model types provided, default use all.")
        models_to_run = ["base", "full_finetuned", "lora"]
    else:
        models_to_run = args.model_types
    for model_type in models_to_run:
        if model_type not in ["base", "full_finetuned", "lora"]:
            print(f"{model_type} not supported.")
            continue
        folder_key = f"{model_type}_output_folder"
        model_key = f"{model_type}_model_path"
        create_output_folders(config[folder_key])
        pipeline = load_model(model_type, config[model_key], config['base_model_path'])
        generate_model_images(model_type, pipeline, config)
        print(f"{model_type.capitalize()} model images: {config[folder_key]}")
        print(f"\nTotal images per prompt for each models: {config['num_images_per_prompt']}")
    
    print("\n================= Generation Complete =================")
    print("\nUse the evaluation script to compare pixelation and aesthetic scores!")

if __name__ == "__main__":
    main()