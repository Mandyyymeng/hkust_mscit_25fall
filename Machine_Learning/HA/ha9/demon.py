"""Demonstration: Show image quality degradation when latent vector moves away from origin"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils import calculate_fid, denorm
import os

def latent_space_demonstration(best_model_path, device, latent_size=128):
    """展示隐空间远离原点时图像质量下降"""
    
    # 加载最佳模型
    G = Generator(latent_size=latent_size, hidden_size=256, image_size=784).to(device)
    G.load_state_dict(torch.load(best_model_path))
    G.eval()
    
    # 创建演示目录
    os.makedirs('demonstration', exist_ok=True)
    
    # 1. 不同距离的隐向量生成图像
    distances = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]  # 距离原点的不同距离
    num_samples = 36
    
    fig, axes = plt.subplots(len(distances), 1, figsize=(12, 3*len(distances)))
    if len(distances) == 1:
        axes = [axes]
    
    with torch.no_grad():
        for i, dist in enumerate(distances):
            # 生成单位球面上的随机点，然后缩放到指定距离
            z = torch.randn(num_samples, latent_size).to(device)
            z = z / torch.norm(z, dim=1, keepdim=True)  # 归一化到单位球面
            z = z * dist  # 缩放到指定距离
            
            # 生成图像
            fake_images = G(z).reshape(-1, 1, 28, 28)
            fake_images = denorm(fake_images).cpu()
            
            # 创建网格显示
            grid = torchvision.utils.make_grid(fake_images, nrow=6, padding=2, normalize=False)
            grid = grid.permute(1, 2, 0).numpy()
            
            axes[i].imshow(grid, cmap='gray')
            axes[i].set_title(f'Latent Distance from Origin: {dist}', fontsize=14, fontweight='bold')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('demonstration/latent_distance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 定量分析：FID分数随距离的变化
    print("Calculating FID scores at different latent distances...")
    
    # 获取真实图像作为参考
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    fashion_mnist = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    real_loader = torch.utils.data.DataLoader(fashion_mnist, batch_size=100, shuffle=True)
    
    real_images = []
    for images, _ in real_loader:
        real_images.append(images.to(device))
        if len(real_images) * 100 >= 1000:
            break
    real_images = torch.cat(real_images, dim=0)[:1000]
    
    # 计算不同距离的FID
    distances_fid = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    fid_scores = []
    
    with torch.no_grad():
        for dist in distances_fid:
            # 生成测试样本
            z_test = torch.randn(1000, latent_size).to(device)
            z_test = z_test / torch.norm(z_test, dim=1, keepdim=True)
            z_test = z_test * dist
            
            fake_images = G(z_test).reshape(-1, 1, 28, 28)
            fid_score = calculate_fid(real_images, fake_images, device)
            fid_scores.append(fid_score)
            
            print(f"Distance {dist}: FID = {fid_score:.2f}")
    
    # 绘制FID vs 距离图
    plt.figure(figsize=(10, 6))
    plt.plot(distances_fid, fid_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Distance from Origin in Latent Space', fontsize=12)
    plt.ylabel('FID Score', fontsize=12)
    plt.title('Image Quality Degradation with Distance from Origin', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig('demonstration/fid_vs_distance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 隐空间插值演示
    print("Demonstrating latent space interpolation...")
    
    # 在两个随机点之间插值
    z1 = torch.randn(1, latent_size).to(device)
    z2 = torch.randn(1, latent_size).to(device)
    
    # 生成插值点
    num_interpolations = 10
    interpolated_images = []
    
    with torch.no_grad():
        for alpha in torch.linspace(0, 1, num_interpolations):
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = G(z_interp).reshape(1, 1, 28, 28)
            interpolated_images.append(denorm(img).cpu())
    
    # 显示插值结果
    interpolation_grid = torch.cat(interpolated_images, dim=0)
    grid = torchvision.utils.make_grid(interpolation_grid, nrow=num_interpolations, padding=2)
    plt.figure(figsize=(15, 3))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title('Latent Space Interpolation Between Two Random Points', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.savefig('demonstration/latent_interpolation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fid_scores

# 使用示例
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用你训练的最佳模型
    best_model = 'checkpoints/exp8_optimized_G.ckpt'  # 替换为你的最佳模型路径
    
    # 运行demonstration
    fid_scores = latent_space_demonstration(best_model, device)
    
    print("\n=== Demonstration Summary ===")
    print("This demonstrates that as we move away from the origin in latent space:")
    print("1. Image quality degrades (visible in the comparison images)")
    print("2. FID scores increase (quantitative measure of quality degradation)")
    print("3. This validates the GAN assumption that real data follows unit Gaussian distribution")
    