import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
import warnings
import requests

warnings.filterwarnings('ignore')


class ImageNetClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_class_index()
        self.setup_models()
        self.setup_transforms()

    def load_class_index(self):
        """Load or download ImageNet class index"""
        if Path('imagenet_class_index.json').exists():
            with open('imagenet_class_index.json', 'r') as f:
                self.class_index = json.load(f)
        else:
            print("📥 Downloading ImageNet class index...")
            url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
            response = requests.get(url)
            self.class_index = response.json()
            with open('imagenet_class_index.json', 'w') as f:
                json.dump(self.class_index, f)
            print("✅ ImageNet class index downloaded and saved")

        self.id_to_class = {int(idx): self.class_index[str(idx)][1] for idx in self.class_index.keys()}

    def setup_models(self):
        """Initialize all 6 pretrained models"""
        print("🔄 Loading 6 pretrained models...")
        self.models = {
            'AlexNet': models.alexnet(pretrained=True), #(weights=models.AlexNet_Weights.IMAGENET1K_V1),
            'VGG16': models.vgg16(pretrained=True),#(weights=models.VGG16_Weights.IMAGENET1K_V1),
            'ResNet50': models.resnet50(pretrained=True),#(weights=models.ResNet50_Weights.IMAGENET1K_V1),
            'Inception V3': models.inception_v3(pretrained=True),#(weights=models.Inception_V3_Weights.IMAGENET1K_V1),
            'DenseNet121': models.densenet121(pretrained=True),#(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
            'MobileNetV2': models.mobilenet_v2(pretrained=True),#(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        }

        for name, model in self.models.items():
            model.to(self.device)
            model.eval()
            print(f"✅ {name} loaded successfully")

    def setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.transform_inception = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_images(self, image_folder, label_folder):
        """Load images and ground truth labels"""
        images = {}
        ground_truth = {}

        # Load original images
        image_extensions = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
        for ext in image_extensions:
            for img_path in Path(image_folder).glob(f'*{ext}'):
                image_id = img_path.stem
                images[image_id] = img_path

        # Load ground truth labels
        for ext in image_extensions:
            for label_path in Path(label_folder).glob(f'*{ext}'):
                filename = label_path.stem
                parts = filename.split('_')
                if len(parts) >= 3:
                    image_id = parts[0]
                    gt_id = parts[1]
                    gt_name = '_'.join(parts[2:])
                    ground_truth[image_id] = {
                        'id': gt_id,
                        'name': gt_name
                    }

        print(f"📁 Loaded {len(images)} images and {len(ground_truth)} ground truth labels")
        return images, ground_truth

    def predict_single_image(self, image_path, model_name):
        """Predict a single image with specified model"""
        try:
            image = Image.open(image_path).convert('RGB')

            if model_name == 'Inception V3':
                transform = self.transform_inception
            else:
                transform = self.transform

            image_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.models[model_name](image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top5_prob, top5_idx = torch.topk(probabilities, 5)

            results = []
            for i in range(5):
                class_id = top5_idx[i].item()
                results.append({
                    'class_id': class_id,
                    'class_name': self.id_to_class[class_id],
                    'probability': top5_prob[i].item()
                })

            return results
        except Exception as e:
            print(f"❌ Error predicting {image_path} with {model_name}: {e}")
            return []

    def evaluate_all_models(self, image_folder, label_folder):
        """Evaluate all 6 models on all images"""
        images, ground_truth = self.load_images(image_folder, label_folder)
        results = []

        print("🚀 Starting evaluation of all 6 models...")
        for image_id, image_path in tqdm(images.items(), desc="Processing images"):
            true_label = ground_truth.get(image_id, {})

            for model_name in self.models.keys():
                predictions = self.predict_single_image(image_path, model_name)
                top_prediction = predictions[0]

                results.append({
                    'image_id': image_id,
                    'model': model_name,
                    'predicted_class': top_prediction['class_name'],
                    'predicted_probability': top_prediction['probability'],
                    'true_class': true_label.get('name', 'Unknown'),
                    'true_id': true_label.get('id', 'Unknown'),
                    'is_correct': top_prediction['class_name'] == true_label.get('name', ''),
                    'top5_predictions': predictions
                })

        return pd.DataFrame(results)


def save_predictions_to_txt(df, output_dir):
    """保存每个图像的预测结果到txt文件"""
    output_file = output_dir / 'detailed_predictions.txt'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("IMAGE CLASSIFICATION PREDICTIONS\n")
        f.write("=" * 80 + "\n\n")

        # 按图像分组
        grouped = df.groupby('image_id')

        for image_id, group in grouped:
            # 获取真实标签（所有行都一样，取第一个）
            true_class = group.iloc[0]['true_class']
            true_id = group.iloc[0]['true_id']

            f.write(f"IMAGE: {image_id}\n")
            f.write(f"TRUE LABEL: {true_class} (ID: {true_id})\n")
            f.write("-" * 50 + "\n")

            # 写入每个模型的预测
            for _, row in group.iterrows():
                status = "✓ CORRECT" if row['is_correct'] else "✗ WRONG"
                f.write(f"{row['model']:15} → {row['predicted_class']:30} "
                        f"[{row['predicted_probability']:.4f}] {status}\n")

            f.write("\n" + "=" * 80 + "\n\n")

    print(f"📝 Detailed predictions saved to: {output_file}")


def create_visualizations(df, output_dir):
    """Create visualization charts with improved styling"""

    # 设置全局样式参数 - 增大字体和调整透明度
    plt.rcParams.update({
        'font.size': 16,  # 增大字体
        'font.weight': 'bold',
        'axes.linewidth': 2,
        'lines.linewidth': 2,
        'patch.linewidth': 1.5
    })

    sns.set_palette("husl")
    sns.set_style("whitegrid")

    # 1. Model Accuracy Comparison - 调整颜色和透明度
    plt.figure(figsize=(12, 8))
    accuracy_by_model = df.groupby('model')['is_correct'].mean().sort_values(ascending=False)

    # 使用更鲜艳的颜色和透明度
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    bars = plt.bar(accuracy_by_model.index, accuracy_by_model.values * 100,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                   width=0.6)  # 减小柱子宽度

    plt.title('Model Accuracy Comparison', fontsize=20, fontweight='bold', pad=20)
    plt.ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
    plt.xlabel('Model', fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontweight='bold')
    plt.ylim(0, 100)

    # 在柱子上添加数值标签
    for bar, value in zip(bars, accuracy_by_model.values * 100):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.1f}%', ha='center', va='bottom',
                 fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_accuracy.png', dpi=300, bbox_inches='tight',
                transparent=True)  # 添加透明背景
    plt.close()

    # 2. Prediction Confidence Distribution - 调整样式
    plt.figure(figsize=(12, 8))
    box_plot = sns.boxplot(data=df, x='model', y='predicted_probability',
                           palette='Set2', width=0.6)  # 减小宽度
    plt.title('Prediction Confidence Distribution\nby Model',
              fontsize=20, fontweight='bold', pad=20)
    plt.ylabel('Prediction Probability', fontsize=18, fontweight='bold')
    plt.xlabel('Model', fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight',
                transparent=True)
    plt.close()

    # 3. Model Agreement Heatmap
    plt.figure(figsize=(10, 8))
    models_list = df['model'].unique()
    model_agreement = []

    for model1 in models_list:
        row = []
        for model2 in models_list:
            if model1 == model2:
                row.append(1.0)
            else:
                df1 = df[df['model'] == model1][['image_id', 'predicted_class']]
                df2 = df[df['model'] == model2][['image_id', 'predicted_class']]
                merged = pd.merge(df1, df2, on='image_id', suffixes=('_1', '_2'))
                agreement = (merged['predicted_class_1'] == merged['predicted_class_2']).mean()
                row.append(agreement)
        model_agreement.append(row)

    sns.heatmap(model_agreement, annot=True, fmt='.2f',
                xticklabels=models_list, yticklabels=models_list,
                cmap='Blues', cbar_kws={'label': 'Rate'},
                annot_kws={'fontweight': 'bold', 'fontsize': 12},
                alpha=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_agreement.png', dpi=300, bbox_inches='tight',
                transparent=True)
    plt.close()

    # 4. Performance by Image - 调整样式
    plt.figure(figsize=(12, 8))
    image_performance = df.groupby('image_id')['is_correct'].mean().sort_values(ascending=False)

    bars = plt.bar(range(len(image_performance)), image_performance.values * 100,
                   color='lightgreen', alpha=0.7, edgecolor='darkgreen',
                   linewidth=1.5, width=0.6)  # 减小宽度

    plt.title('Per Image Performance', fontsize=20, fontweight='bold', pad=20)
    plt.ylabel('Correct Ratio (%)', fontsize=18, fontweight='bold')
    plt.xlabel('Image ID', fontsize=18, fontweight='bold')
    plt.xticks(range(len(image_performance)), image_performance.index,
               rotation=45, ha='right', fontweight='bold')
    plt.ylim(0, 100)

    # 添加数值标签
    for bar, value in zip(bars, image_performance.values * 100):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.0f}%', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_image.png', dpi=300, bbox_inches='tight',
                transparent=True)
    plt.close()


def generate_analysis_report(df, output_dir):
    """Generate comprehensive analysis report and save to file"""

    report_content = []
    report_content.append("=" * 80)
    report_content.append("IMAGE CLASSIFICATION MODEL COMPARISON REPORT")
    report_content.append("=" * 80)

    # Overall Statistics
    total_images = df['image_id'].nunique()
    total_predictions = len(df)
    models_used = len(df['model'].unique())

    report_content.append(f"\n📊 OVERALL STATISTICS")
    report_content.append(f"   Total Images: {total_images}")
    report_content.append(f"   Total Predictions: {total_predictions}")
    report_content.append(f"   Models Evaluated: {models_used}")

    # Model Performance Ranking
    report_content.append(f"\n🏆 MODEL PERFORMANCE RANKING")
    performance = df.groupby('model').agg({
        'is_correct': ['mean', 'sum', 'count'],
        'predicted_probability': 'mean'
    }).round(3)

    performance.columns = ['accuracy', 'correct_count', 'total_predictions', 'avg_confidence']
    performance = performance.sort_values('accuracy', ascending=False)

    performance_table = performance.reset_index()
    report_content.append(performance_table.to_string(index=False))

    # Save report to file
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))

    print(f"📄 Analysis report saved to: {report_path}")
    return performance


def main():
    # Create output directory
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)

    print("🔬 ImageNet Classification Analysis")
    print("Evaluating 6 models: AlexNet, VGG16, ResNet50, Inception V3, DenseNet121, MobileNetV2")

    # Initialize classifier
    classifier = ImageNetClassifier()

    # Evaluate all models
    results_df = classifier.evaluate_all_models('images', 'images_label')

    # 保存详细的预测结果到txt
    print("📝 Saving detailed predictions...")
    save_predictions_to_txt(results_df, output_dir)

    # Generate visualizations
    print("📊 Creating visualizations...")
    create_visualizations(results_df, output_dir)

    # Generate analysis report
    print("📄 Generating analysis report...")
    performance_stats = generate_analysis_report(results_df, output_dir)

    print(f"\n✅ Analysis completed successfully!")
    print(f"📁 All results saved in: {output_dir}/")
    print(f"📈 Visualizations: 4 charts saved as PNG files")
    print(f"📋 Reports: analysis_report.txt and detailed_predictions.txt")


if __name__ == "__main__":
    main()