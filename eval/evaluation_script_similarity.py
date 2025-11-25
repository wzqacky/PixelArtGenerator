import os
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import numpy as np

class ModelFolderMultiPromptEvaluator:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        """
        Initialize evaluator for three dedicated model folders
        - Explicit model-folder mapping
        - Parses prompt index from filenames
        - Supports your 10 specific prompts
        :param clip_model_name: CLIP model name (see options in docstring)
        """
        # Auto-detect GPU/CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load CLIP model and processor
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.model.eval()
        
        # Define your 10 prompts (index 0-9 matches prompt_{0-9} in filenames)
        self.prompts = [
            "fantasy castle, retro game style, vibrant colors, simple background",
            "landscape with mountains and rivers, top-down view, minimal color palette",
            "warrior with sword, side profile, simple background, bold design",
            "city skyline at night, neon lights, sharp edges, no blur",
            "fox in a forest, cute style, vibrant colors, clean design",
            "space scene with planets and stars, retro arcade style, high contrast",
            "burger and fries, flat design, vibrant colors, simple composition",
            "dungeon interior, top-down, torch light, minimal details",
            "mermaid in ocean, colorful scales, simple background",
            "car race, retro game style, dynamic pose, bold lines"
        ]
        print(f"Loaded {len(self.prompts)} prompts (index 0-9)")

    def parse_prompt_index_from_filename(self, filename):
        """
        Parse prompt index from filename (ignores model prefix, since folder defines model)
        Filename format: {any_prefix}_prompt_{idx}_seed_{...}.png
        :param filename: Image filename (e.g., "full_finetuned_prompt_0_seed_1042.png")
        :return: prompt_idx (int) or None if parsing fails
        """
        # Regex pattern to extract prompt index (ignores model prefix)
        pattern = r"_prompt_(\d+)_seed_\d+\.png$"
        match = re.search(pattern, filename)
        
        if not match:
            print(f"Warning: Could not parse prompt index from filename: {filename} (invalid format)")
            return None
        
        prompt_idx = int(match.group(1))
        # Validate prompt index (0-9 for your 10 prompts)
        if prompt_idx < 0 or prompt_idx >= len(self.prompts):
            print(f"Warning: Invalid prompt index {prompt_idx} in filename: {filename}")
            return None
        
        return prompt_idx

    def load_images_from_model_folders(self, model_folder_config):
        """
        Load images from three dedicated model folders
        :param model_folder_config: Dictionary with {model_name: folder_path}
        :return: List of tuples (image, model_name, prompt_text, prompt_idx, image_path)
        """
        image_metadata_list = []
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        
        print(f"\nLoading images from {len(model_folder_config)} model folders...")
        for model_name, folder_path in model_folder_config.items():
            print(f"\nProcessing model: {model_name} (folder: {folder_path})")
            
            # Validate folder exists
            if not os.path.isdir(folder_path):
                print(f"Warning: Folder not found for {model_name}: {folder_path} - skipping")
                continue
            
            # Load all valid images in the folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(supported_formats):
                    image_path = os.path.join(folder_path, filename)
                    
                    # Parse prompt index from filename
                    prompt_idx = self.parse_prompt_index_from_filename(filename)
                    if prompt_idx is None:
                        continue
                    
                    prompt_text = self.prompts[prompt_idx]
                    
                    # Load image (convert to RGB)
                    try:
                        img = Image.open(image_path).convert("RGB")
                        image_metadata_list.append((img, model_name, prompt_text, prompt_idx, image_path))
                        print(f"Loaded: {filename} → Prompt #{prompt_idx}")
                    except Exception as e:
                        print(f"Failed to load {image_path}: {str(e)}")
        
        print(f"\nSuccessfully loaded {len(image_metadata_list)} valid images across all models")
        return image_metadata_list

    def compute_batch_similarity(self, images, prompts):
        """
        Compute similarity scores for a batch of (image, prompt) pairs
        :param images: List of PIL Image objects
        :param prompts: List of prompt texts (one per image)
        :return: List of similarity scores (0-1 range)
        """
        if not images or len(images) != len(prompts):
            return []
        
        # Preprocess images and prompts for CLIP
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Fast inference (no gradient computation)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # FIX: Use correct attribute names (image_embeds → not image_embeddings)
            image_embeds = outputs.image_embeds  # Corrected line
            text_embeds = outputs.text_embeds    # Corrected line
            
            # Normalize embeddings for cosine similarity
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Cosine similarity → normalize to 0-1 range
            similarity = (image_embeds * text_embeds).sum(dim=-1)
            scores = ((similarity + 1) / 2).cpu().numpy().flatten()  # Map [-1,1] → [0,1]
        
        return scores.tolist()

    def evaluate_models(self, model_folder_config, batch_size=8, save_results=True, output_dir="model_folder_evaluation"):
        """
        Evaluate all models using their dedicated folders
        :param model_folder_config: Dictionary with {model_name: folder_path}
        :param batch_size: Batch size for CLIP inference (adjust for GPU memory)
        :param save_results: Whether to save CSV results
        :param output_dir: Directory to save results/plots
        :return: Detailed results, model summary, prompt summary, cross summary
        """
        # Create output directory
        if save_results and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Step 1: Load images and metadata
        image_metadata = self.load_images_from_model_folders(model_folder_config)
        if not image_metadata:
            print("Error: No valid images found for evaluation")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Step 2: Compute similarity scores in batches
        print("\nStarting similarity score computation...")
        all_results = []
        total_batches = (len(image_metadata) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_metadata))
            batch = image_metadata[start_idx:end_idx]
            
            # Extract batch components
            batch_images = [item[0] for item in batch]
            batch_models = [item[1] for item in batch]
            batch_prompts = [item[2] for item in batch]
            batch_prompt_idxs = [item[3] for item in batch]
            batch_paths = [item[4] for item in batch]
            
            # Compute scores for the batch
            batch_scores = self.compute_batch_similarity(batch_images, batch_prompts)
            
            # Store results
            for model, prompt, prompt_idx, img_path, score in zip(
                batch_models, batch_prompts, batch_prompt_idxs, batch_paths, batch_scores
            ):
                all_results.append({
                    "model_name": model,
                    "prompt_index": prompt_idx,
                    "prompt_text": prompt,
                    "image_path": img_path,
                    "similarity_score": round(score, 4)
                })
            
            print(f"Completed batch {batch_idx + 1}/{total_batches}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Step 3: Save detailed results
        if save_results and not results_df.empty:
            detailed_csv = os.path.join(output_dir, "detailed_evaluation_results.csv")
            results_df.to_csv(detailed_csv, index=False)
            print(f"\nDetailed results saved to: {detailed_csv}")
        
        # Step 4: Generate summary statistics
        # Model-level summary (average across all prompts)
        model_summary = results_df.groupby("model_name").agg({
            "similarity_score": ["mean", "std", "max", "min", "count"]
        }).round(4)
        model_summary.columns = ["avg_score", "std_dev", "max_score", "min_score", "total_images"]
        model_summary = model_summary.reset_index()
        
        # Prompt-level summary (average across all models)
        prompt_summary = results_df.groupby(["prompt_index", "prompt_text"]).agg({
            "similarity_score": ["mean", "std", "count"]
        }).round(4)
        prompt_summary.columns = ["avg_score", "std_dev", "total_images"]
        prompt_summary = prompt_summary.reset_index().sort_values("prompt_index")
        
        # Model-Prompt cross summary (performance per model per prompt)
        cross_summary = results_df.groupby(["model_name", "prompt_index", "prompt_text"]).agg({
            "similarity_score": "mean"
        }).round(4).reset_index()
        cross_summary = cross_summary.pivot(
            index="model_name", 
            columns="prompt_index", 
            values="similarity_score"
        ).fillna(0)  # Fill 0 for missing model-prompt combinations
        
        # Save summaries
        if save_results:
            model_summary.to_csv(os.path.join(output_dir, "model_performance_summary.csv"), index=False)
            prompt_summary.to_csv(os.path.join(output_dir, "prompt_performance_summary.csv"), index=False)
            cross_summary.to_csv(os.path.join(output_dir, "model_prompt_cross_summary.csv"))
            print("Summary files saved successfully")
        
        # Print key results
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        
        print("\n1. Model Overall Performance (Average Across All Prompts):")
        print(model_summary.to_string(index=False))
        
        print("\n2. Prompt Average Performance (Across All Models):")
        print(prompt_summary[["prompt_index", "avg_score", "total_images"]].to_string(index=False))
        
        return results_df, model_summary, prompt_summary, cross_summary

    def generate_visualizations(self, model_summary, prompt_summary, cross_summary, results_df, output_dir="model_folder_evaluation"):
        """
        Generate 4-panel visualization of results
        """
        if model_summary.empty or prompt_summary.empty:
            print("Warning: Cannot generate visualizations - insufficient data")
            return
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.dpi'] = 150
        
        # Create 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle("Text-to-Image Alignment Evaluation (Three Model Folders)", fontsize=16, fontweight='bold')
        
        # Color palette
        model_colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(model_summary)]
        prompt_colors = plt.cm.viridis(np.linspace(0, 1, len(prompt_summary)))
        
        # Plot 1: Model Overall Performance (Bar Chart)
        ax1 = axes[0, 0]
        models = model_summary["model_name"]
        avg_scores = model_summary["avg_score"]
        std_devs = model_summary["std_dev"]
        
        bars = ax1.bar(models, avg_scores, yerr=std_devs, capsize=8, color=model_colors, alpha=0.8, edgecolor='black')
        ax1.set_title("Model Overall Performance\n(Higher = Better Alignment)", fontweight='bold')
        ax1.set_ylabel("Average Similarity Score (0-1)")
        ax1.set_ylim(0, 1.05)
        
        # Add value labels
        for bar, score in zip(bars, avg_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{score:.4f}", ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Prompt Average Performance (Bar Chart)
        ax2 = axes[0, 1]
        prompt_labels = [f"Prompt {idx}" for idx in prompt_summary["prompt_index"]]
        prompt_avg_scores = prompt_summary["avg_score"]
        
        bars = ax2.bar(prompt_labels, prompt_avg_scores, color=prompt_colors, alpha=0.8, edgecolor='black')
        ax2.set_title("Prompt Average Performance\n(Across All Models)", fontweight='bold')
        ax2.set_ylabel("Average Similarity Score (0-1)")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, prompt_avg_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{score:.4f}", ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Model-Prompt Cross Performance (Heatmap)
        ax3 = axes[1, 0]
        im = ax3.imshow(cross_summary.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax3.set_title("Model-Prompt Cross Performance Heatmap", fontweight='bold')
        ax3.set_xlabel("Prompt Index")
        ax3.set_ylabel("Model Name")
        ax3.set_xticks(range(len(cross_summary.columns)))
        ax3.set_xticklabels([f"{int(col)}" for col in cross_summary.columns])
        ax3.set_yticks(range(len(cross_summary.index)))
        ax3.set_yticklabels(cross_summary.index)
        
        # Add value annotations
        for i in range(len(cross_summary.index)):
            for j in range(len(cross_summary.columns)):
                value = cross_summary.iloc[i, j]
                ax3.text(j, i, f"{value:.3f}", ha='center', va='center', 
                        color='black' if value < 0.7 else 'white', fontsize=8)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label("Similarity Score (0-1)")
        
        # Plot 4: Score Distribution by Model (Box Plot)
        ax4 = axes[1, 1]
        score_distributions = [
            results_df[results_df["model_name"] == model]["similarity_score"].values
            for model in model_summary["model_name"]
        ]
        
        bp = ax4.boxplot(score_distributions, labels=model_summary["model_name"], patch_artist=True)
        for patch, color in zip(bp['boxes'], model_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax4.set_title("Score Distribution by Model", fontweight='bold')
        ax4.set_ylabel("Similarity Score (0-1)")
        ax4.set_ylim(0, 1.0)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "model_evaluation_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Visualizations saved to: {plot_path}")

# -----------------------------------------------------------------------------
# Usage Example (UPDATE THESE PATHS TO MATCH YOUR FOLDERS!)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Critical Configuration: Map model names to your three folders
    MODEL_FOLDER_CONFIG = {
        "Base Model": "./base_model_images",        # Path to your base model images
        "Full Finetune": "./full_finetuned_images", # Path to your full finetune images
        "LoRA": "./lora_model_images"               # Path to your LoRA model images
    }
    
    # 2. CLIP Model Configuration (adjust based on GPU memory)
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # Fast (2GB VRAM)
    # CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"  # More accurate (8GB+ VRAM)
    
    # 3. Inference Batch Size (reduce if OOM errors)
    BATCH_SIZE = 8
    
    # 4. Initialize evaluator
    evaluator = ModelFolderMultiPromptEvaluator(clip_model_name=CLIP_MODEL_NAME)
    
    # 5. Run evaluation
    detailed_results, model_summary, prompt_summary, cross_summary = evaluator.evaluate_models(
        model_folder_config=MODEL_FOLDER_CONFIG,
        batch_size=BATCH_SIZE,
        save_results=True,
        output_dir="model_folder_evaluation"
    )
    
    # 6. Generate visualizations (if results exist)
    if not detailed_results.empty:
        evaluator.generate_visualizations(
            model_summary=model_summary,
            prompt_summary=prompt_summary,
            cross_summary=cross_summary,
            results_df=detailed_results,
            output_dir="model_folder_evaluation"
        )