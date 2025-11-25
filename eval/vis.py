import matplotlib.pyplot as plt
import numpy as np

# Set plot style for better readability
plt.style.use('default')

# Experimental data extracted from evaluation results
models = ['Base Model', 'Full Finetune', 'LoRA Finetune']
aesthetic_scores = [48.76, 48.81, 48.58]  # Aesthetic quality score
pixel_scores = [39.79, 33.57, 37.24]      # Pixel-art style adherence score
similarity_scores = [63.82, 65.90, 66.16] # Image-text alignment score

# Create a 1x3 subplot layout with appropriate figure size
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# fig.suptitle('StableDiffusion Fine-tuned Models: Pixel Art Generation Performance', 
#              fontsize=5, fontweight='bold', y=0.98)

# Define consistent color palette for model distinction
colors = ['#2E86AB', '#A23B72', '#F18F01']

# --------------------------
# Subplot 1: Aesthetic Score
# --------------------------
axes[0].bar(models, aesthetic_scores, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
axes[0].set_title('Aesthetic Quality Score', fontsize=14, fontweight='semibold', pad=20)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_ylim(48.4, 48.9)  # Narrow y-axis range to highlight minor differences
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on top of bars
for i, score in enumerate(aesthetic_scores):
    axes[0].text(i, score + 0.01, f'{score:.2f}', ha='center', va='bottom', 
                 fontsize=11, fontweight='medium')

# --------------------------
# Subplot 2: Pixel-art Score
# --------------------------
axes[1].bar(models, pixel_scores, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
axes[1].set_title('Pixel-art Style Adherence', fontsize=14, fontweight='semibold', pad=20)
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_ylim(32, 41)  # Optimize y-axis range for clear comparison
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on top of bars
for i, score in enumerate(pixel_scores):
    axes[1].text(i, score + 0.4, f'{score:.2f}', ha='center', va='bottom', 
                 fontsize=11, fontweight='medium')

# --------------------------
# Subplot 3: Image-Text Similarity
# --------------------------
axes[2].bar(models, similarity_scores, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
axes[2].set_title('Image-Text Alignment Score', fontsize=14, fontweight='semibold', pad=20)
axes[2].set_ylabel('Score', fontsize=12)
axes[2].set_ylim(63, 67)  # Optimize y-axis range for clear comparison
axes[2].grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on top of bars
for i, score in enumerate(similarity_scores):
    axes[2].text(i, score + 0.2, f'{score:.2f}', ha='center', va='bottom', 
                 fontsize=11, fontweight='medium')

# --------------------------
# Global layout adjustments
# --------------------------
# Adjust spacing between subplots
plt.tight_layout()
# Adjust top margin to prevent title cutoff
plt.subplots_adjust(top=0.88)

# Save plot with high resolution (300 DPI) for publication/presentation
plt.savefig('pixel_art_model_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('pixel_art_model_comparison.pdf', dpi=300, bbox_inches='tight')  # PDF format for vector quality

# Display the plot
plt.show()

# Print summary statistics for quick reference
print("="*60)
print("Model Performance Summary (English)")
print("="*60)
print(f"{'Model':<15} {'Aesthetic Score':<18} {'Pixel-art Score':<18} {'Image-Text Score':<18}")
print("-"*60)
for i in range(len(models)):
    print(f"{models[i]:<15} {aesthetic_scores[i]:<18.2f} {pixel_scores[i]:<18.2f} {similarity_scores[i]:<18.2f}")
print("="*60)