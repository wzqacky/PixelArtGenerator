import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from scipy import ndimage
import cv2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class PixelArtEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.setup_models()
        
    def setup_models(self):
        """设置用于评估的模型"""
        # 加载预训练的ResNet用于美学评估
        self.aesthetic_model = models.resnet50(pretrained=True)
        self.aesthetic_model.fc = nn.Linear(self.aesthetic_model.fc.in_features, 1)
        self.aesthetic_model.eval()
        self.aesthetic_model.to(self.device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def calculate_color_simplicity(self, image_array):
        """计算颜色简洁度 - 像素艺术通常使用有限颜色"""
        # 转换为RGB数组
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        
        # 计算唯一颜色数量
        unique_colors = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0)
        color_count = len(unique_colors)
        
        # 像素艺术通常颜色较少，理想范围是2-32种颜色
        if color_count <= 8:
            color_score = 1.0
        elif color_count <= 16:
            color_score = 0.8
        elif color_count <= 32:
            color_score = 0.6
        elif color_count <= 64:
            color_score = 0.4
        else:
            color_score = 0.2
            
        return color_score, color_count
    
    def calculate_edge_sharpness(self, image_array):
        """计算边缘锐度 - 像素艺术应该有清晰的边缘"""
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # 使用拉普拉斯算子计算边缘
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 归一化到0-1范围
        sharpness_score = min(laplacian_var / 1000.0, 1.0)
        return sharpness_score, laplacian_var
    
    def calculate_pixelation_level(self, image_array):
        """计算像素化程度"""
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # 计算高频内容（像素艺术应该有明显的高频内容）
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        
        # 计算高频能量比例
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # 高频区域（外部区域）
        high_freq_energy = np.sum(magnitude_spectrum[center_h-10:center_h+10, center_w-10:center_w+10])
        total_energy = np.sum(magnitude_spectrum)
        
        if total_energy > 0:
            high_freq_ratio = high_freq_energy / total_energy
        else:
            high_freq_ratio = 0
        
        pixelation_score = min(high_freq_ratio * 5, 1.0)
        return pixelation_score, high_freq_ratio
    
    def calculate_contrast(self, image_array):
        """计算对比度"""
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # 计算对比度（标准差）
        contrast = np.std(gray)
        contrast_score = min(contrast / 64.0, 1.0)  # 归一化
        return contrast_score, contrast
    
    def aesthetic_assessment(self, image):
        """美学评估（简化版）"""
        try:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                aesthetic_score = torch.sigmoid(self.aesthetic_model(input_tensor))
            return aesthetic_score.item()
        except:
            # 如果模型评估失败，使用基于规则的替代方案
            return self.rule_based_aesthetic(image)
    
    def rule_based_aesthetic(self, image):
        """基于规则的美学评估替代方案"""
        image_array = np.array(image)
        
        # 基于颜色分布、对比度、锐度等综合评估
        color_score, _ = self.calculate_color_simplicity(image_array)
        sharpness_score, _ = self.calculate_edge_sharpness(image_array)
        contrast_score, _ = self.calculate_contrast(image_array)
        
        # 综合评分
        aesthetic_score = (color_score * 0.3 + sharpness_score * 0.4 + contrast_score * 0.3)
        return aesthetic_score
    
    def evaluate_single_image(self, image_path):
        """评估单张图片"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # 计算各项指标
            color_score, color_count = self.calculate_color_simplicity(image_array)
            sharpness_score, sharpness_value = self.calculate_edge_sharpness(image_array)
            pixelation_score, pixelation_ratio = self.calculate_pixelation_level(image_array)
            contrast_score, contrast_value = self.calculate_contrast(image_array)
            aesthetic_score = self.aesthetic_assessment(image)
            
            # 综合像素化评分（权重调整）
            pixelation_final_score = (
                color_score * 0.25 +
                sharpness_score * 0.35 +
                pixelation_score * 0.40
            )
            
            return {
                'pixelation_score': pixelation_final_score,
                'aesthetic_score': aesthetic_score,
                'color_score': color_score,
                'color_count': color_count,
                'sharpness_score': sharpness_score,
                'sharpness_value': sharpness_value,
                'pixelation_ratio': pixelation_ratio,
                'contrast_score': contrast_score,
                'contrast_value': contrast_value
            }
        except Exception as e:
            print(f"Error evaluating {image_path}: {str(e)}")
            return None

def batch_evaluate_folders(evaluator, folders_dict, output_csv='evaluation_results.csv'):
    """批量评估文件夹中的图片"""
    results = []
    
    for model_name, folder_path in folders_dict.items():
        print(f"Evaluating {model_name}...")
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist, skipping...")
            continue
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            print(f"No images found in {folder_path}")
            continue
        
        model_results = []
        for i, image_file in enumerate(image_files):
            print(f"  Processing {i+1}/{len(image_files)}: {image_file}")
            image_path = os.path.join(folder_path, image_file)
            
            result = evaluator.evaluate_single_image(image_path)
            if result is not None:
                result['model'] = model_name
                result['image_file'] = image_file
                model_results.append(result)
        
        if model_results:
            # 计算该模型的平均分
            avg_pixelation = np.mean([r['pixelation_score'] for r in model_results])
            avg_aesthetic = np.mean([r['aesthetic_score'] for r in model_results])
            
            print(f"{model_name} - Avg Pixelation: {avg_pixelation:.3f}, Avg Aesthetic: {avg_aesthetic:.3f}")
            results.extend(model_results)
    
    # 保存结果
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        
        # 生成统计报告
        generate_statistical_report(df, folders_dict)
        
    return results

def generate_statistical_report(df, folders_dict):
    """生成统计分析报告"""
    print("\n" + "="*50)
    print("STATISTICAL REPORT")
    print("="*50)
    
    # 按模型分组统计
    stats = df.groupby('model').agg({
        'pixelation_score': ['mean', 'std', 'min', 'max'],
        'aesthetic_score': ['mean', 'std', 'min', 'max'],
        'color_count': ['mean', 'std'],
        'sharpness_value': ['mean', 'std'],
        'contrast_value': ['mean', 'std']
    }).round(4)
    
    print(stats)
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 像素化评分分布
    plt.subplot(2, 3, 1)
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.hist(model_data['pixelation_score'], alpha=0.7, label=model, bins=20)
    plt.xlabel('Pixelation Score')
    plt.ylabel('Frequency')
    plt.title('Pixelation Score Distribution')
    plt.legend()
    
    # 美学评分分布
    plt.subplot(2, 3, 2)
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.hist(model_data['aesthetic_score'], alpha=0.7, label=model, bins=20)
    plt.xlabel('Aesthetic Score')
    plt.ylabel('Frequency')
    plt.title('Aesthetic Score Distribution')
    plt.legend()
    
    # 平均分比较
    plt.subplot(2, 3, 3)
    avg_scores = df.groupby('model')[['pixelation_score', 'aesthetic_score']].mean()
    x = np.arange(len(avg_scores))
    width = 0.35
    
    plt.bar(x - width/2, avg_scores['pixelation_score'], width, label='Pixelation', alpha=0.8)
    plt.bar(x + width/2, avg_scores['aesthetic_score'], width, label='Aesthetic', alpha=0.8)
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Average Scores by Model')
    plt.xticks(x, avg_scores.index)
    plt.legend()
    
    # 颜色数量分布
    plt.subplot(2, 3, 4)
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.hist(model_data['color_count'], alpha=0.7, label=model, bins=20)
    plt.xlabel('Color Count')
    plt.ylabel('Frequency')
    plt.title('Color Count Distribution')
    plt.legend()
    
    # 散点图：像素化 vs 美学
    plt.subplot(2, 3, 5)
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['pixelation_score'], model_data['aesthetic_score'], 
                   alpha=0.6, label=model, s=50)
    plt.xlabel('Pixelation Score')
    plt.ylabel('Aesthetic Score')
    plt.title('Pixelation vs Aesthetic Score')
    plt.legend()
    
    # 模型排名
    plt.subplot(2, 3, 6)
    final_scores = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        # 综合评分（可根据需求调整权重）
        combined_score = (model_data['pixelation_score'].mean() * 0.6 + 
                         model_data['aesthetic_score'].mean() * 0.4)
        final_scores.append((model, combined_score))
    
    final_scores.sort(key=lambda x: x[1], reverse=True)
    models_ranked = [x[0] for x in final_scores]
    scores_ranked = [x[1] for x in final_scores]
    
    plt.bar(models_ranked, scores_ranked, color=['gold', 'silver', 'brown'])
    plt.xlabel('Model')
    plt.ylabel('Combined Score')
    plt.title('Model Ranking (Pixelation 60% + Aesthetic 40%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('evaluation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印排名
    print("\n" + "="*30)
    print("MODEL RANKING")
    print("="*30)
    for i, (model, score) in enumerate(final_scores, 1):
        print(f"{i}. {model}: {score:.4f}")

def main():
    # 定义要评估的文件夹
    folders_dict = {
        'base_model': 'base_model_images',
        'full_finetuned': 'full_finetuned_model_images', 
        'lora_model': 'lora_model_images'
    }
    
    # 初始化评估器
    evaluator = PixelArtEvaluator()
    
    # 批量评估
    print("Starting batch evaluation...")
    results = batch_evaluate_folders(evaluator, folders_dict)
    
    if results:
        print("\nEvaluation completed successfully!")
        print(f"Total images evaluated: {len(results)}")
    else:
        print("No results were generated. Please check your folder paths and image files.")

if __name__ == "__main__":
    main()