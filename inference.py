import sys
from pathlib import Path
import argparse
import json
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline,\
            StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

from eval.utils import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pixel art images from text prompts.")
    parser.add_argument("--model_path", type=str, help="Path to the fine-tuned full model/ LoRA weight/ Controlnet.")
    parser.add_argument("--base_model_path", type=str, help="Path to the base model.")
    parser.add_argument("--model_type", type=str, default="base", choices=["base", "full_finetuned", "lora", "controlnet"])
    parser.add_argument("--prompt", type=str, required=True, help="The text prompt to generate an image.")
    parser.add_argument("--control_image", type=str, help="The control image for controlnet image generation.")
    parser.add_argument("--negative_prompt", type=str, default="3d, smooth, blurry, soft, intricate, artifacts, low-quality, ", help="The negative text prompt to avoid in the image.")
    parser.add_argument("--output_path", type=str, default="output_images", help="Path to the output images.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for generation.")
    parser.add_argument("--seed", type=int, help="Seed for random noise generation.")
    args = parser.parse_args()

    if args.model_type == "base" or args.model_type == "full_finetuned":
        if not args.model_path and not args.base_model_path:
            print("Error: You must provide either --model_path or --base_model_path.")
            sys.exit(1)
        model_path = args.model_path if args.model_path else args.base_model_path
    else:
        if not args.model_path or not args.base_model_path:
            print("Error: You must provide --model_path and --base_model_path.")
            sys.exit(1)
        model_path = args.model_path

    pipe = load_model(args.model_type, model_path, args.base_model_path)
    generator = None
    if args.seed:
        generator = torch.Generator(pipe.device).manual_seed(args.seed)

    generation_kwargs = {
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
        "image": load_image(args.control_image).resize((512,512)) if args.model_type == "controlnet" else None}
    generated_image = pipe(
        args.prompt,
        **generation_kwargs
    ).images[0]
    output_path = Path(args.output_path) / Path(model_path).name
    output_path.mkdir(exist_ok=True)
    output_path = output_path / f"{args.prompt.replace(' ','-')}.png"
    generated_image.save(output_path)
