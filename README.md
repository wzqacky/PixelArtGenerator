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
The training script requires diffusers==0.36.0.dev0, which has to be built from source
```
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e ".[torch]"
```
Then, run the training
```
bash scripts/train.sh
```

# Inference
## Base model
```
python inference.py --base_model stabilityai/stable-diffusion-xl-base-1.0 --prompt "a cute shiba inu"
```
## Full-finetuned model
```
python inference.py --model_path checkpoints/pixel-art-model_sdxl-filtered-dataset --prompt "a cute shiba inu"
```
## LoRA model
```
python inference.py --model_path checkpoints/pixel-art-model_sdxl_lora-filterted-dataset --base_model stabilityai/stable-diffusion-xl-base-1.0 --lora --prompt "a cute shiba inu"
```

# Evaluation
## Generate data
Base model
```
python generation_script_v2.py --model base
```

Full-finetuned model
```
python generation_script_v2.py --model full_finetuned
```

LoRA model
```
python generation_script_v2.py --model lora
```

## Make evaluation
```
python evaluation_script_v2.py 
```
