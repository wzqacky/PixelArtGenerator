#!/bin/bash
export MODEL_NAME="stabilityai/stable-diffusion-"
export train_data_dir="data/"
export output_dir="pixel-art-model"

python train_text_to_image.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$train_data_dir \
    --image_column image \
    --caption_column caption \
    --use_ema \
    --resolution=512 \
    --center_crop \
    --random_flip \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --mixed_precision="fp16" \
    --max_train_steps=16000 \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --checkpointing_steps 5000\
    --validation_prompt "A cute shiba inu"\
    --validation_epochs 1\
    --num_validation_images 1\
    --output_dir=$output_dir
