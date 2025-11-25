from pathlib import Path
from diffusers import DiffusionPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
import torch
from typing import Dict, List
import csv

def create_output_folders(output_path: str) -> None:
    """Create output folder if it doesn't exist (no overwite warning—safe for new generation)"""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"Output folder initialized: {output_path}")

def load_model(model_type: str, model_path: str, base_model_path: str) -> DiffusionPipeline:
    """Load model based on type (base/full_finetuned/LoRA) using config paths"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {model_type} model to {device}...")
    
    try:
        if model_type == "base" or model_type == "full_finetuned":
            pipe = DiffusionPipeline.from_pretrained(model_path)
        elif model_type == "lora":
            pipe = DiffusionPipeline.from_pretrained(base_model_path)
            pipe.load_lora_weights(model_path)
        elif model_type == "controlnet":
            controlnet = ControlNetModel.from_pretrained(model_path)
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Optimize for speed/memory
        pipe.to(device)
        if device == "cuda":
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing()
            
        print(f"Successfully loaded {model_type} model into {device}")
        return pipe
        
    except Exception as e:
        print(f"Error loading {model_type} model: {str(e)}")
        raise SystemExit(1)
    
def load_prompts_from_csv(csv_path: str) -> Dict[str, List[str]]:
    """Loads prompts from a CSV file with 'base' and 'specialized' columns."""
    prompts = {"base": [], "specialized": []}
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('base'):
                    prompts["base"].append(row['base'])
                if row.get('specialized'):
                    prompts["specialized"].append(row['specialized'])
        if not prompts["base"] or not prompts["specialized"]:
            raise ValueError("CSV file is empty or missing 'base' or 'specialized' columns.")
        return prompts
    except FileNotFoundError:
        print(f"Error: Prompt CSV file not found at '{csv_path}'")
        raise SystemExit(1)
    except Exception as e:
        print(f"Error reading prompts from CSV: {e}")
        raise SystemExit(1)
