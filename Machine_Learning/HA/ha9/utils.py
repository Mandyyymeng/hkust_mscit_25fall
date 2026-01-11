import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm
import logging
import os
from PIL import Image
import matplotlib.pyplot as plt

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def denorm(x):
    """反标准化，将图像从[-1,1]转换到[0,1]"""
    out = (x + 1) / 2
    return out.clamp(0, 1)

class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, image_size, activation='relu'):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(latent_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, image_size)
        self.tanh = nn.Tanh()
        
        # 可配置的激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x

# InceptionV3特征提取器
class InceptionV3FeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # 创建Inception网络模型
        try:
            self.model = models.inception_v3(pretrained=True, transform_input=False)
            self.model.aux_logits = False
            if hasattr(self.model, 'AuxLogits'):
                self.model.AuxLogits = None
            self.model.fc = nn.Identity()
        except Exception as e:
            print(f"使用兼容性初始化: {e}")
            self.model = models.inception_v3(pretrained=True)
            self.model.aux_logits = False
            self.model.AuxLogits = None
            self.model.fc = nn.Identity()
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("InceptionV3特征提取器初始化完成")

    def extract_features_from_tensor(self, images):
        """从张量中提取特征"""
        features = []
        batch_size = 64
        
        # 确保图像是3通道
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        num_batches = (len(images) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(images))
                
                batch_images = images[start_idx:end_idx].to(self.device)
                
                # 调整图像尺寸为299x299
                if batch_images.shape[2] != 299 or batch_images.shape[3] != 299:
                    batch_images = torch.nn.functional.interpolate(
                        batch_images, size=(299, 299), mode='bilinear', align_corners=False
                    )
                
                # 应用标准化
                batch_images = self._normalize_batch(batch_images)
                
                # 提取特征
                batch_features = self.model(batch_images).detach().cpu().numpy()
                features.append(batch_features)
        
        return np.concatenate(features, axis=0)

    def _normalize_batch(self, batch):
        """标准化图像批次"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        return (batch - mean) / std

# 全局特征提取器实例
fid_extractor = InceptionV3FeatureExtractor(device)

def calculate_fid(real_images, fake_images, device, max_samples=1000):
    """计算FID分数"""
    if len(real_images) > max_samples:
        real_images = real_images[:max_samples]
    if len(fake_images) > max_samples:
        fake_images = fake_images[:max_samples]
    
    print(f"提取真实图像特征... ({len(real_images)} 张)")
    real_features = fid_extractor.extract_features_from_tensor(real_images)
    
    print(f"提取生成图像特征... ({len(fake_images)} 张)")
    fake_features = fid_extractor.extract_features_from_tensor(fake_images)
    
    print(f"特征提取完成: 真实{real_features.shape[0]}个, 生成{fake_features.shape[0]}个")
    
    # 计算统计量
    real_mean = np.mean(real_features, axis=0)
    fake_mean = np.mean(fake_features, axis=0)
    
    real_cov = np.cov(real_features, rowvar=False)
    fake_cov = np.cov(fake_features, rowvar=False)
    
    # 计算FID
    mean_diff = np.sum((real_mean - fake_mean) ** 2)
    
    covmean, _ = sqrtm(real_cov.dot(fake_cov), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid_score = mean_diff + np.trace(real_cov + fake_cov - 2 * covmean)
    
    if fid_score < 0:
        fid_score = 0.0
    
    return fid_score

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """计算WGAN-GP的梯度惩罚"""
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def save_loss_curves(results, exp_name, save_dir):
    """保存损失曲线"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['epoch_d_losses'], label='Discriminator Loss')
    plt.plot(results['epoch_g_losses'], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{exp_name} - Epoch Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    steps = min(1000, len(results['d_losses']))
    plt.plot(results['d_losses'][-steps:], label='D Loss', alpha=0.7)
    plt.plot(results['g_losses'][-steps:], label='G Loss', alpha=0.7)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(f'{exp_name} - Step Loss (last {steps} steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_sample_images(generator, latent_size, num_images=64, device='cpu'):
    """生成样本图像"""
    with torch.no_grad():
        z = torch.randn(num_images, latent_size).to(device)
        fake_images = generator(z).reshape(-1, 1, 28, 28)
    return fake_images

def load_model(model, model_path, device='cpu'):
    """加载模型权重"""
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return True
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {str(e)}")
        return False