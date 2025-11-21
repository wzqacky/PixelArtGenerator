export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export train_data_dir="data/palette_filtered_pixelart_images.parquet"
export output_dir="checkpoints/pixel-art-model_controlnet-palette-filtered-dataset"

python training/train_controlnet_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --pretrained_vae_model_name_or_path=$VAE_NAME \
    --train_data_dir=$train_data_dir \
    --output_dir=$output_dir \
    --enable_xformers_memory_efficient_attention \
    --image_column image --image_column_type byte --caption_column caption \
    --resolution=512 \
    --learning_rate=1e-5 \
    --checkpointing_steps=5000 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=40000 \
    --use_8bit_adam \
    --validation_image "controlnet_validation/palette/conditioning_image_1.png" "controlnet_validation/palette/conditioning_image_2.png" \
    --validation_prompt "a cyberpunk cityscape filled with skyscrapers at serene night" "a cute shiba inu" \
    --validation_steps 4000\
    --num_validation_images 1\
    --report_to="wandb" \
    --seed=42 \
    --save_intermediate_latents