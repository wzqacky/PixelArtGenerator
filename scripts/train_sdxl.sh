#!/bin/bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export train_data_dir="data/filtered_pixelart_images.parquet"
export output_dir="pixel-art-model_sdxl-filtered-dataset"

python training/train_text_to_image_sdxl.py \
    --report_to wandb \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --pretrained_vae_model_name_or_path=$VAE_NAME \
    --train_data_dir=$train_data_dir \
    --enable_xformers_memory_efficient_attention \
    --image_column image \
    --image_column_type byte \
    --caption_column caption \
    --resolution=512 --center_crop --random_flip \
    --proportion_empty_prompts=0.2 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --mixed_precision="fp16" \
    --max_train_steps=40000 \
    --use_8bit_adam \
    --learning_rate=1e-06 \
    --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --checkpointing_steps 5000\
    --validation_prompt "A cute shiba inu"\
    --validation_epochs 2\
    --num_validation_images 3\
    --save_intermediate_latents\
    --output_dir=$output_dir
