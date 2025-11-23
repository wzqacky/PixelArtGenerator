import sys
from pathlib import Path
import argparse
import json
from diffusers import DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline,\
            StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pixel art images from text prompts.")
    parser.add_argument("--model_path", type=str, help="Path to the full model/ LoRA weight.")
    parser.add_argument("--base_model", type=str, help="Path to the base model.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--lora", action="store_true", help="Using LoRA model weight or not.")
    parser.add_argument("--controlnet", action="store_true", help="Using controlnet weight or not.")
    parser.add_argument("--prompt", type=str, required=True, help="The text prompt to generate an image.")
    parser.add_argument("--control_image", type=str, help="The control image for controlnet image generation.")
    parser.add_argument("--negative_prompt", type=str, default="3d, smooth, blurry, soft, intricate, artifacts, low-quality, ", help="The negative text prompt to avoid in the image.")
    parser.add_argument("--output_path", type=str, default="output_images", help="Path to the output images.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for generation.")
    parser.add_argument("--seed", type=int, help="Seed for random noise generation.")
    args = parser.parse_args()

    if not args.model_path and not args.checkpoint_path and not args.base_model:
        print("Error: You must provide either --model_path or --checkpoint_path or --base_model")
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.base_model and not args.model_path:
        model_path = args.base_model
        pipe = DiffusionPipeline.from_pretrained(model_path)
    if args.model_path:
        model_path = args.model_path
        if args.lora:
            if not args.base_model:
                print("Error: Please provide the base model for loading the LoRA weight.")
                sys.exit(1)
            pipe = DiffusionPipeline.from_pretrained(args.base_model)
            pipe.load_lora_weights(model_path)
        elif args.controlnet:
            if not args.base_model:
                print("Error: Please provide the base model for loading the controlnet weight.")
                sys.exit(1)
            if not args.control_image:
                print("Error: Please provide the control image for controlnet image generation.")
            controlnet = ControlNetModel.from_pretrained(model_path)
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(args.base_model, controlnet=controlnet)
            control_image = load_image(args.control_image).resize((512,512))
        else:
            pipe = DiffusionPipeline.from_pretrained(model_path)
            
    elif args.checkpoint_path:
        model_path = args.checkpoint_path
        unet = UNet2DConditionModel.from_pretrained(args.checkpoint_path + "unet", torch_dtype=torch.float16)
        with open(args.checkpoint_path + 'model.index.json', 'r') as f:
            model_index = json.load(f)
        base_model = model_index.get('_name_or_path', "")
        pipe = StableDiffusionPipeline.from_pretrained(base_model, unet=unet, torch_dtype=torch.float16)
    pipe.to(device)
    print(f"Pipeline loaded into device: {device}")

    generator = None
    if args.seed:
        generator = torch.Generator(device).manual_seed(args.seed)

    generation_kwargs = {
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
        "image": control_image if args.controlnet else None}
    generated_image = pipe(
        args.prompt,
        **generation_kwargs
    ).images[0]
    output_path = Path(args.output_path) / Path(model_path).name
    output_path.mkdir(exist_ok=True)
    output_path = output_path / f"{args.prompt.replace(' ','-')}.png"
    generated_image.save(output_path)
