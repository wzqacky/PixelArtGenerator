# Installing dependencies
For Nvidia, install stable pytorch with the following versions compatible with CUDA from https://pytorch.org/get-started/locally/, e.g. CUDA 12.8:

`pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128`

Install the dependencies:

`pip install -r requirements.txt`

The training scripts requires `diffusers==0.36.0.dev0`, which has to be built from source:
```
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e ".[torch]"
```

# Downloading dataset
## Raw dataset
- Huggingface: https://huggingface.co/datasets/bghira/free-to-use-pixelart
- Kaggle: https://www.kaggle.com/datasets/artvandaley/curated-pixel-art-512x512
```
bash scripts/download_raw_dataset.sh
```

## Processed dataset
- https://huggingface.co/datasets/wzqacky/captioned_pixelart_images
```
bash scripts/download_processed_dataset.sh
```

## Data Preprocessing
Generate captions for images in `data/pixilart_processed` and `data/pixel-art-512x512`, using BLIP2: `Salesforce/blip2-opt-6.7b`, and also filter
the wordings related to "pixel-art"
```
bash scripts/preprocess.sh --upload_to_hf
```

# Training
## Full finetuning
```
bash scripts/train_sdxl.sh
```
## LoRA
```
bash scripts/train_sdxl_lora.sh
```
## ControlNet
```
bash scripts/train_canny_controlnet.sh
bash scripts/train_palette_controlnet.sh
```
## Sbatch
```
sbatch --wait -o slurm.out scripts/run.sbatch
```

# Inference
## Base model
```
python inference.py \
    --base_model stabilityai/stable-diffusion-xl-base-1.0 \
    --prompt "a cute shiba inu" \
    --model_type "base"
```
## Full-finetuned model
```
python inference.py \
    --model_path wzqacky/pixel-art-model-sdxl \
    --prompt "a cute shiba inu" \
    --model_type "full_finetuned"
```
## LoRA model
```
python inference.py \
    --model_path wzqacky/pixel-art-model-sdxl-lora \
    --base_model stabilityai/stable-diffusion-xl-base-1.0 \
    --prompt "a cute shiba inu" \
    --model_type "lora"
```
## Controlnet
```
python inference.py \
    --model_path wzqacky/pixel-art-model-controlnet-canny \
    --base_model stabilityai/stable-diffusion-xl-base-1.0 \
    --prompt "a cute shiba inu" \
    --control_image controlnet_validation/canny/a-cute-cartoon-shiba-inu.png \
    --model_type "controlnet"
```
```
python inference.py \
    --model_path wzqacky/pixel-art-model-controlnet-palette \
    --base_model stabilityai/stable-diffusion-xl-base-1.0 \
    --prompt "a cute shiba inu" \
    --control_image controlnet_validation/palette/a-cute-cartoon-shiba-inu.png \
    --model_type "controlnet"
```
# Uploading finetuned model
```
python hf/upload_to_hf.py \
    --path checkpoints/pixel-art-model_sdxl-filtered-dataset \
    --repo_id wzqacky/pixel-art-model-sdxl \
    --repo_type model
```

# Evaluation
## Generate data (T2I)
Change the evaluation parameters at `generation_config.yaml` and the prompts at `prompts.csv`
Base model
```
cd eval/T2I
python generation_script.py --model base
```

Full-finetuned model
```
python generation_script.py --model full_finetuned
```

LoRA model
```
python generation_script.py --model lora
```

## Make evaluation
```
python evaluation_script.py 
```