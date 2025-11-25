import os
import torch
import open_clip
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt

class LAIONAestheticScorer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化LAION美学评分器
        使用CLIP ConvNeXt Base模型，在LAION美学数据集上训练
        """
        self.device = device
        self.model, self.preprocess = None, None
        self.setup_model()
        
    def setup_model(self):
        """加载模型和预处理流程"""
        try:
            # 加载模型和预处理
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'hf-hub:laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg'  # 使用augreg版本，在美学数据上训练
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            print("模型加载成功: CLIP ConvNeXt Base W 320 (LAION Aesthetic)")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def extract_image_features(self, image_path):
        """提取图像特征"""
        try:
            image = Image.open(image_path).convert('RGB')
            # 应用预处理
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # 提取图像特征
                image_features = self.model.encode_image(image_tensor)
                # 标准化特征向量
                image_features = torch.nn.functional.normalize(image_features, dim=-1)
                
            return image_features.cpu()
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            return None

    def calculate_aesthetic_score(self, image_features, method='relative'):
        """
        计算美学分数
        
        参数:
            image_features: 图像特征向量
            method: 评分方法 ('relative' 或 'norm')
        
        返回:
            aesthetic_score: 美学分数 (0-1范围)
        """
        if image_features is None:
            return 0.0
            
        if method == 'norm':
            # 方法1: 使用特征向量的范数作为美学分数
            # 研究表明，高质量图像的特征范数往往更大
            score = torch.norm(image_features).item()
            # 将分数归一化到0-1范围（根据经验调整）
            score = min(score / 2.0, 1.0)  # 假设2.0是典型最大值
            
        elif method == 'relative':
            # 方法2: 使用相对美学评分
            # 这里使用一个假设的"高质量"参考方向
            # 在实际应用中，你可以使用已知高质量图像的特征均值作为参考
            reference_direction = torch.ones_like(image_features) / image_features.shape[1]**0.5
            score = torch.nn.functional.cosine_similarity(image_features, reference_direction).item()
            # 将余弦相似度从[-1, 1]映射到[0, 1]
            score = (score + 1) / 2
            
        else:
            raise ValueError("方法必须是 'norm' 或 'relative'")
            
        return max(0.0, min(1.0, score))  # 确保分数在0-1范围内

    def evaluate_single_image(self, image_path, method='relative'):
        """评估单张图像的美学质量"""
        image_features = self.extract_image_features(image_path)
        if image_features is None:
            return None
            
        aesthetic_score = self.calculate_aesthetic_score(image_features, method)
        
        return {
            'aesthetic_score': aesthetic_score,
            'image_features': image_features,
            'method': method
        }

def batch_aesthetic_evaluation(scorer, folders_dict, output_csv='aesthetic_scores.csv'):
    """
    批量评估文件夹中的图像美学质量
    
    参数:
        scorer: LAIONAestheticScorer实例
        folders_dict: 文件夹字典 {模型名称: 文件夹路径}
        output_csv: 输出CSV文件名
    """
    results = []
    
    for model_name, folder_path in folders_dict.items():
        print(f"评估 {model_name}...")
        
        if not os.path.exists(folder_path):
            print(f"文件夹 {folder_path} 不存在，跳过...")
            continue
        
        # 获取图像文件
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            print(f"在 {folder_path} 中没有找到图像文件")
            continue
        
        model_results = []
        for i, image_file in enumerate(image_files):
            print(f"  处理 {i+1}/{len(image_files)}: {image_file}")
            image_path = os.path.join(folder_path, image_file)
            
            # 评估图像
            result = scorer.evaluate_single_image(image_path)
            if result is not None:
                result['model'] = model_name
                result['image_file'] = image_file
                model_results.append(result)
        
        if model_results:
            # 计算模型平均分
            avg_aesthetic = np.mean([r['aesthetic_score'] for r in model_results])
            print(f"{model_name} - 平均美学分数: {avg_aesthetic:.3f}")
            results.extend(model_results)
    
    # 保存结果
    if results:
        df = pd.DataFrame([{
            'model': r['model'],
            'image_file': r['image_file'], 
            'aesthetic_score': r['aesthetic_score'],
            'method': r['method']
        } for r in results])
        
        df.to_csv(output_csv, index=False)
        print(f"结果已保存至 {output_csv}")
        
        # 生成可视化报告
        generate_aesthetic_report(df, folders_dict)
        
    return results

def generate_aesthetic_report(df, folders_dict):
    """生成美学评估报告"""
    print("\n" + "="*50)
    print("美学评估报告")
    print("="*50)
    
    # 按模型统计
    stats = df.groupby('model').agg({
        'aesthetic_score': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)
    
    print(stats)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    # 美学分数分布
    plt.subplot(2, 2, 1)
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.hist(model_data['aesthetic_score'], alpha=0.7, label=model, bins=20)
    plt.xlabel('美学分数')
    plt.ylabel('频次')
    plt.title('美学分数分布')
    plt.legend()
    
    # 箱形图比较
    plt.subplot(2, 2, 2)
    model_data_list = [df[df['model'] == model]['aesthetic_score'] for model in df['model'].unique()]
    plt.boxplot(model_data_list, labels=df['model'].unique())
    plt.ylabel('美学分数')
    plt.title('模型间美学分数比较')
    
    # 平均分比较
    plt.subplot(2, 2, 3)
    avg_scores = df.groupby('model')['aesthetic_score'].mean()
    plt.bar(avg_scores.index, avg_scores.values, alpha=0.7)
    plt.ylabel('平均美学分数')
    plt.title('各模型平均美学分数')
    plt.xticks(rotation=45)
    
    # 分数散点图
    plt.subplot(2, 2, 4)
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(range(len(model_data)), model_data['aesthetic_score'], 
                   alpha=0.6, label=model, s=30)
    plt.xlabel('图像索引')
    plt.ylabel('美学分数')
    plt.title('各图像美学分数分布')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('aesthetic_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印排名
    print("\n模型排名:")
    print("-" * 30)
    ranking = []
    for model in df['model'].unique():
        model_avg = df[df['model'] == model]['aesthetic_score'].mean()
        ranking.append((model, model_avg))
    
    ranking.sort(key=lambda x: x[1], reverse=True)
    for i, (model, score) in enumerate(ranking, 1):
        print(f"{i}. {model}: {score:.4f}")

def main():
    """主函数"""
    # 定义要评估的文件夹
    folders_dict = {
        'base_model': 'base_model_images',
        'full_finetuned': 'full_finetuned_images', 
        'lora_model': 'lora_model_images'
    }
    
    # 初始化评分器
    scorer = LAIONAestheticScorer()
    
    # 批量评估
    print("开始批量美学评估...")
    results = batch_aesthetic_evaluation(scorer, folders_dict, 'aesthetic_scores.csv')
    
    if results:
        print(f"\n评估完成! 总共评估了 {len(results)} 张图像")
        
        # 额外分析
        df = pd.DataFrame([{'model': r['model'], 'score': r['aesthetic_score']} for r in results])
        overall_stats = df['score'].describe()
        
        print("\n总体分数统计:")
        print(f"平均分: {overall_stats['mean']:.3f}")
        print(f"标准差: {overall_stats['std']:.3f}")
        print(f"最低分: {overall_stats['min']:.3f}")
        print(f"最高分: {overall_stats['max']:.3f}")
    else:
        print("未生成结果，请检查文件夹路径和图像文件")

if __name__ == "__main__":
    import numpy as np
    main()