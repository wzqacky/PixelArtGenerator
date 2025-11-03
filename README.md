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