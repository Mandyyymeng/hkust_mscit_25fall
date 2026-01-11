"""测试文件 - 直接从exp_config获取配置，加载权重评估FID"""
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import logging
import json
import numpy as np
import argparse
import sys

# 添加当前目录到路径，以便导入配置文件
sys.path.append('.')

# 从配置文件导入实验配置
from exp_config import experiments, base_config
from utils import *

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
setup_logging()

def get_config_from_model_path(model_path):
    """从模型文件名提取实验名称并获取配置"""
    filename = os.path.basename(model_path)
    
    # 提取实验名称 (exp10_gradient_penalty_G_final.pth -> exp10_gradient_penalty)
    if '_G_final.pth' in filename:
        exp_name = filename.replace('_G_final.pth', '')
    elif '_G_epoch_' in filename:
        exp_name = filename.split('_G_epoch_')[0]
    else:
        exp_name = filename.replace('.pth', '')
    
    logging.info(f"从文件名提取实验名称: {exp_name}")
    
    # 从exp_config获取配置
    if exp_name in experiments:
        # 合并基础配置和实验特定配置
        config = base_config.copy()
        config.update(experiments[exp_name])
        logging.info(f"找到实验配置: {exp_name}")
        return config, exp_name
    else:
        # 如果找不到实验配置，使用基础配置
        logging.warning(f"未找到实验 {exp_name} 的配置，使用基础配置")
        return base_config.copy(), exp_name

def load_model_from_checkpoint(model_path, model_class, config, device='cpu'):
    """从检查点加载模型"""
    model = model_class(
        latent_size=config['latent_size'],
        hidden_size=config['hidden_size'],
        image_size=config['image_size'],
        activation=config.get('activation', 'relu')
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logging.info(f"成功加载模型: {model_path}")
        logging.info(f"模型配置: latent_size={config['latent_size']}, hidden_size={config['hidden_size']}")
        return model
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

def evaluate_fid(generator, config, num_samples=1000, save_samples=True):
    """评估生成器的FID分数"""
    # 加载真实数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root='./data/', train=True, transform=transform, download=True
    )
    
    # 获取真实图像
    logging.info(f"收集 {num_samples} 张真实图像...")
    real_images = []
    for i in range(min(num_samples, len(fashion_mnist))):
        img, _ = fashion_mnist[i]
        real_images.append(img)
    real_images = torch.stack(real_images).to(device)
    
    # 生成伪造图像
    logging.info(f"生成 {num_samples} 张伪造图像...")
    with torch.no_grad():
        z = torch.randn(num_samples, config['latent_size']).to(device)
        fake_images = generator(z).reshape(-1, 1, 28, 28)
    
    # 保存样本图像
    if save_samples:
        os.makedirs('test_samples', exist_ok=True)
        save_image(denorm(fake_images[:64]), 'test_samples/generated_samples.png')
        save_image(denorm(real_images[:64]), 'test_samples/real_samples.png')
        logging.info("样本图像已保存到 test_samples/ 目录")
    
    # 计算FID
    logging.info("计算FID分数...")
    fid_score = calculate_fid(real_images, fake_images, device)
    
    return fid_score

def evaluate_single_model(model_path, num_samples=1000):
    """评估单个模型文件"""
    logging.info(f"评估模型: {model_path}")
    
    # 从模型文件名获取配置
    config, exp_name = get_config_from_model_path(model_path)
    
    try:
        # 加载生成器
        generator = load_model_from_checkpoint(model_path, Generator, config, device)
        
        # 评估FID
        fid_score = evaluate_fid(generator, config, num_samples)
        
        result = {
            'model_path': model_path,
            'exp_name': exp_name,
            'fid_score': fid_score,
            'config': config,
            'num_samples': num_samples
        }
        
        logging.info(f"模型 {exp_name} 的FID分数: {fid_score:.4f}")
        return result
        
    except Exception as e:
        logging.error(f"评估模型失败: {str(e)}")
        return None

def evaluate_all_models(model_dir='saved_models', num_samples=1000):
    """评估所有模型"""
    results = {}
    
    # 获取所有模型文件
    if not os.path.exists(model_dir):
        logging.error(f"模型目录不存在: {model_dir}")
        return results
    
    # 查找所有生成器模型文件
    model_files = []
    for file in os.listdir(model_dir):
        if file.endswith('_G_final.pth'):
            model_files.append(os.path.join(model_dir, file))
    
    logging.info(f"找到 {len(model_files)} 个模型文件")
    
    for model_path in model_files:
        result = evaluate_single_model(model_path, num_samples)
        if result:
            results[result['exp_name']] = result
    
    # 保存结果
    with open('fid_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 输出总结
    logging.info("\n" + "="*50)
    logging.info("FID评估总结")
    logging.info("="*50)
    
    for exp_name, result in results.items():
        logging.info(f"{exp_name:30} | FID: {result['fid_score']:8.4f}")
    
    # 找到最佳模型
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['fid_score'])
        logging.info(f"\n最佳模型: {best_model[0]}")
        logging.info(f"最佳FID: {best_model[1]['fid_score']:.4f}")
        
    # 保存结果到详细文件
    detailed_results = {}
    for exp_name, result in results.items():
        detailed_results[exp_name] = {
            'exp_name': exp_name,
            'model_path': result['model_path'],
            'fid_score': result['fid_score'],
            'config': result['config'],
            'num_samples': result['num_samples']
        }
    
    # 保存详细结果
    with open('detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估训练好的GAN模型')
    parser.add_argument('--model_path', type=str, help='单个模型路径')
    parser.add_argument('--eval_all', action='store_true', help='评估所有实验')
    parser.add_argument('--num_samples', type=int, default=1000, help='用于评估的样本数量')
    parser.add_argument('--model_dir', type=str, default='saved_models', help='模型目录')
    
    args = parser.parse_args()
    
    if args.model_path:
        # 评估特定模型文件
        result = evaluate_single_model(args.model_path, args.num_samples)
        if result:
            print(f"\n评估结果:")
            print(f"实验名称: {result['exp_name']}")
            print(f"模型路径: {result['model_path']}")
            print(f"FID分数: {result['fid_score']:.4f}")
            print(f"样本数量: {result['num_samples']}")
            print(f"配置: latent_size={result['config']['latent_size']}, hidden_size={result['config']['hidden_size']}")
    
    elif args.eval_all:
        # 评估所有模型
        results = evaluate_all_models(args.model_dir, args.num_samples)
        print(f"\n完成了 {len(results)} 个实验的评估")
        print("详细结果保存在: fid_evaluation_results.json")
    
    else:
        print("请指定要评估的模型路径")
        print("用法示例:")
        print("  python test.py --model saved_models/exp10_gradient_penalty_G_final.pth")
        print("  python test.py --eval_all")

if __name__ == "__main__":
    main()