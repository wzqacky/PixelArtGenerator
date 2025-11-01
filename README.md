# Downloading dataset
## Raw dataset
- Huggingface: https://huggingface.co/datasets/bghira/free-to-use-pixelart
- Kaggle: https://www.kaggle.com/datasets/artvandaley/curated-pixel-art-512x512
```
bash scripts/preprocess.sh
```
# Data Proprocessing
We need to separately generate captions for `data/pixilart_processed` and `data/pixel-art-512x512`, using BLIP2: `Salesforce/blip2-opt-6.7b`
```
python preprocess/caption.py --image_dir data/pixilart_processed --output_dir data/
```
Then, we filter wordings related to 'pixel art' in `data/caption.csv`
```
python preprocess/filtered_caption.py --input_csv data/caption.csv --output_csv data/filtered_caption.csv
```
# Training
```

```