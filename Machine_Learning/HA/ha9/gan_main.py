"""主训练文件 - 包含GAN稳定技术和Loss曲线可视化，统一模型保存"""
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
from exp_config import experiments, base_config
from utils import *

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
setup_logging()

# 创建统一的模型保存目录
MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------- 改进的模型结构 --------------------------
class Discriminator(nn.Module):
    def __init__(self, image_size, hidden_size, leaky=0.2, activation='leaky_relu', use_spectral_norm=False):
        super(Discriminator, self).__init__()
        
        # 谱归一化包装器
        linear1 = nn.Linear(image_size, hidden_size)
        linear2 = nn.Linear(hidden_size, hidden_size)
        linear3 = nn.Linear(hidden_size, 1)
        
        if use_spectral_norm:
            self.linear1 = nn.utils.spectral_norm(linear1)
            self.linear2 = nn.utils.spectral_norm(linear2)
            self.linear3 = nn.utils.spectral_norm(linear3)
        else:
            self.linear1 = linear1
            self.linear2 = linear2
            self.linear3 = linear3
        
        # 可配置的激活函数
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(leaky)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.LeakyReLU(leaky)
            
        # WGAN不使用sigmoid
        self.use_sigmoid = not use_spectral_norm

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x

# 梯度惩罚函数
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """计算梯度惩罚"""
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

def analyze_latent_space(generator, device, latent_size, num_samples=64):
    """分析隐空间 - 正常采样 vs 远离原点的采样"""
    with torch.no_grad():
        # 正常采样 (μ=0, σ=1)
        z_normal = torch.randn(num_samples, latent_size).to(device)
        images_normal = generator(z_normal).reshape(-1, 1, 28, 28)
        
        # 远离原点的采样 (μ=0, σ=3)
        z_far = torch.randn(num_samples, latent_size).to(device) * 3.0
        images_far = generator(z_far).reshape(-1, 1, 28, 28)
    
    return images_normal, images_far

def plot_loss_curves(results, exp_name, sample_dir):
    """绘制Loss曲线"""
    plt.figure(figsize=(12, 4))
    
    # 每轮平均Loss
    plt.subplot(1, 2, 1)
    plt.plot(results['epoch_d_losses'], label='Discriminator Loss')
    plt.plot(results['epoch_g_losses'], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{exp_name} - Epoch Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 每步Loss（最后1000步）
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
    plt.savefig(os.path.join(sample_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------- 模型加载函数 --------------------------
def load_trained_model(exp_name, epoch='final', device='cpu'):
    """加载训练好的模型"""
    if epoch == 'final':
        g_path = os.path.join(MODEL_DIR, f'{exp_name}_G_final.pth')
        d_path = os.path.join(MODEL_DIR, f'{exp_name}_D_final.pth')
    else:
        g_path = os.path.join(MODEL_DIR, f'{exp_name}_G_epoch_{epoch}.pth')
        d_path = os.path.join(MODEL_DIR, f'{exp_name}_D_epoch_{epoch}.pth')
    
    # 从实验结果获取配置
    with open('all_experiment_results.json', 'r') as f:
        all_results = json.load(f)
    
    if exp_name not in all_results:
        raise ValueError(f"Experiment {exp_name} not found in results")
    
    config = all_results[exp_name]['config']
    
    # 创建模型实例
    G = Generator(
        latent_size=config['latent_size'],
        hidden_size=config['hidden_size'],
        image_size=config['image_size'],
        activation=config.get('activation', 'relu')
    ).to(device)
    
    D = Discriminator(
        image_size=config['image_size'],
        hidden_size=config['hidden_size'],
        use_spectral_norm=config.get('use_spectral_norm', False)
    ).to(device)
    
    # 加载权重
    G.load_state_dict(torch.load(g_path, map_location=device))
    D.load_state_dict(torch.load(d_path, map_location=device))
    
    G.eval()
    D.eval()
    
    logging.info(f"Loaded models for {exp_name} from epoch {epoch}")
    return G, D, config

def train_experiment(exp_name, config, reload=False):
    """训练单个实验"""
    logging.info(f"Starting experiment: {exp_name}")
    if reload:
        logging.info(f"Reload mode: will try to resume from latest checkpoint")
    logging.info(f"Config: {config}")
    
    # 合并配置
    cfg = base_config.copy()
    cfg.update(config)
    description = cfg.pop('description', 'No description')
    
    # ========== 首先检查是否已经有final模型 ==========
    final_g_path = os.path.join(MODEL_DIR, f'{exp_name}_G_final.pth')
    final_d_path = os.path.join(MODEL_DIR, f'{exp_name}_D_final.pth')

    if os.path.exists(final_g_path) and os.path.exists(final_d_path):
        logging.info(f"Final models already exist for {exp_name}, skipping this experiment")
        return {
            'description': description,
            'config': cfg,
            'fid_score': 0,  # 或者从之前的结果加载
            'final_d_loss': 0,
            'final_g_loss': 0,
            'model_paths': {
                'generator': final_g_path,
                'discriminator': final_d_path
            }
        }
    
    # 创建目录
    sample_dir = f'samples/{exp_name}'
    os.makedirs(sample_dir, exist_ok=True)
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root='./data/', train=True, transform=transform, download=True
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset=fashion_mnist, batch_size=cfg['batch_size'], shuffle=True
    )
    
    # 初始化模型
    activation = cfg.get('activation', 'relu')
    G = Generator(
        latent_size=cfg['latent_size'], 
        hidden_size=cfg['hidden_size'], 
        image_size=cfg['image_size'],
        activation=activation
    ).to(device)
    
    D = Discriminator(
        image_size=cfg['image_size'],
        hidden_size=cfg['hidden_size'],
        leaky=cfg.get('disc_leaky_slope', 0.2),
        activation='leaky_relu' if activation == 'relu' else activation,
        use_spectral_norm=cfg.get('use_spectral_norm', False)
    ).to(device)
    
    # 损失函数
    loss_function = cfg['loss_function']
    if loss_function == 'bce':
        criterion = nn.BCELoss()
    elif loss_function == 'mse':
        criterion = nn.MSELoss()
    else:  # wasserstein
        criterion = None  # WGAN使用原始输出
    
    # 优化器
    d_lr = cfg.get('d_lr', cfg['lr'])
    g_lr = cfg.get('g_lr', cfg['lr'])
    
    d_optimizer = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
    
    # ========== 新增：重载检查点功能 ==========
    start_epoch = 0
    if reload:
        # 查找该实验的最大epoch检查点
        max_epoch = 0
        checkpoint_path = None
        
        logging.info(f"Searching for checkpoints for experiment: {exp_name}")
        
        for file in os.listdir(MODEL_DIR):
            # 打印所有找到的相关文件
            if exp_name in file:
                logging.info(f"Found related file: {file}")
                
            if file.startswith(f'{exp_name}_G_epoch_') and file.endswith('.pth'):
                try:
                    # 从文件名提取epoch数字：exp14_gradient_penalty_improved_G_epoch_20.pth -> 20
                    epoch_num = int(file.split('_epoch_')[1].replace('.pth', ''))
                    logging.info(f"Found checkpoint: {file} -> epoch {epoch_num}")
                    
                    if epoch_num > max_epoch:
                        max_epoch = epoch_num
                        checkpoint_path = file
                        logging.info(f"New max epoch: {max_epoch}")
                except (ValueError, IndexError) as e:
                    logging.warning(f"Failed to parse epoch from {file}: {e}")
                    continue
        
        logging.info(f"Final max epoch found: {max_epoch}, checkpoint: {checkpoint_path}")
        
        if checkpoint_path and max_epoch > 0:
            # 加载生成器
            g_path = os.path.join(MODEL_DIR, checkpoint_path)
            logging.info(f"Loading generator from: {g_path}")
            G.load_state_dict(torch.load(g_path, map_location=device))
            
            # 加载判别器
            d_path = os.path.join(MODEL_DIR, checkpoint_path.replace('_G_', '_D_'))
            logging.info(f"Loading discriminator from: {d_path}")
            if os.path.exists(d_path):
                D.load_state_dict(torch.load(d_path, map_location=device))
            else:
                logging.warning(f"Discriminator checkpoint not found: {d_path}")
            
            # 加载优化器状态（如果存在）
            optimizer_path = os.path.join(MODEL_DIR, checkpoint_path.replace('_G_epoch_', '_optimizer_epoch_'))
            if os.path.exists(optimizer_path):
                logging.info(f"Loading optimizer state from: {optimizer_path}")
                checkpoint = torch.load(optimizer_path, map_location=device)
                d_optimizer.load_state_dict(checkpoint['d_optimizer'])
                g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            else:
                logging.info("No optimizer state found, using fresh optimizers")
            
            start_epoch = max_epoch
            logging.info(f"Successfully reloaded from epoch {start_epoch}")
            
            if start_epoch >= cfg['num_epochs']:
                logging.info(f"Training already completed (epoch {start_epoch}/{cfg['num_epochs']}), skipping to evaluation")
                return _evaluate_experiment(exp_name, config, G, D, data_loader, device, sample_dir, description)
            
        else:
            logging.info("No checkpoint found, starting from scratch")
    
    logging.info(f"Discriminator LR: {d_lr}, Generator LR: {g_lr}")
    logging.info(f"Starting from epoch: {start_epoch}")
    
    # 训练记录（用于Loss曲线）
    results = {
        'd_losses': [], 'g_losses': [], 
        'real_scores': [], 'fake_scores': [],
        'epoch_d_losses': [], 'epoch_g_losses': []  # 每轮的平均loss
    }
    
    n_critic = cfg.get('n_critic', 1)
    use_gp = cfg.get('use_gradient_penalty', False)
    
    for epoch in range(start_epoch, cfg['num_epochs']):
        epoch_d_losses = []
        epoch_g_losses = []
        
        for i, (images, _) in enumerate(data_loader):
            images = images.reshape(images.size(0), -1).to(device)
            batch_size = images.size(0)
            
            # 训练判别器
            d_loss_total = 0
            for _ in range(n_critic):
                # 真实图像
                real_outputs = D(images)
                
                # 伪造图像
                z = torch.randn(batch_size, cfg['latent_size']).to(device)
                fake_images = G(z)
                fake_outputs = D(fake_images.detach())
                
                if loss_function == 'wasserstein':
                    # WGAN损失
                    d_loss_real = -torch.mean(real_outputs)
                    d_loss_fake = torch.mean(fake_outputs)
                    d_loss = d_loss_real + d_loss_fake
                    
                    # 梯度惩罚
                    if use_gp:
                        gp = compute_gradient_penalty(D, images, fake_images, device)
                        d_loss += cfg.get('gp_weight', 10.0) * gp
                else:
                    # 传统GAN损失
                    real_labels = torch.ones(batch_size, 1).to(device)
                    fake_labels = torch.zeros(batch_size, 1).to(device)
                    
                    d_loss_real = criterion(real_outputs, real_labels)
                    d_loss_fake = criterion(fake_outputs, fake_labels)
                    d_loss = d_loss_real + d_loss_fake
                
                # 优化判别器
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                
                d_loss_total += d_loss.item()
            
            d_loss_avg = d_loss_total / n_critic
            epoch_d_losses.append(d_loss_avg)
            
            # 训练生成器
            z = torch.randn(batch_size, cfg['latent_size']).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            
            if loss_function == 'wasserstein':
                g_loss = -torch.mean(outputs)  # WGAN生成器损失
            else:
                target = torch.ones(batch_size, 1).to(device)
                g_loss = criterion(outputs, target)
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            epoch_g_losses.append(g_loss.item())
            
            # 记录结果
            if (i+1) % 200 == 0:
                results['d_losses'].append(d_loss_avg)
                results['g_losses'].append(g_loss.item())
                results['real_scores'].append(real_outputs.mean().item())
                results['fake_scores'].append(fake_outputs.mean().item())
                
                logging.info(f'Epoch [{epoch+1}/{cfg["num_epochs"]}], Step [{i+1}/{len(data_loader)}], '
                            f'd_loss: {d_loss_avg:.4f}, g_loss: {g_loss.item():.4f}, '
                            f'D(x): {real_outputs.mean().item():.2f}, D(G(z)): {fake_outputs.mean().item():.2f}')
        
        # 记录每轮平均loss
        results['epoch_d_losses'].append(np.mean(epoch_d_losses))
        results['epoch_g_losses'].append(np.mean(epoch_g_losses))
        
        # 每10轮保存一次模型检查点
        if (epoch + 1) % 10 == 0:
            torch.save(G.state_dict(), os.path.join(MODEL_DIR, f'{exp_name}_G_epoch_{epoch+1}.pth'))
            torch.save(D.state_dict(), os.path.join(MODEL_DIR, f'{exp_name}_D_epoch_{epoch+1}.pth'))
            
            # 保存优化器状态
            optimizer_state = {
                'd_optimizer': d_optimizer.state_dict(),
                'g_optimizer': g_optimizer.state_dict()
            }
            torch.save(optimizer_state, os.path.join(MODEL_DIR, f'{exp_name}_optimizer_epoch_{epoch+1}.pth'))
            
            logging.info(f"Saved checkpoint at epoch {epoch+1}")
        
        # 保存样本图像
        with torch.no_grad():
            if (epoch + 1) == 1 or (reload and epoch == start_epoch):
                real_images_save = images.reshape(images.size(0), 1, 28, 28)
                save_image(denorm(real_images_save[:64]), 
                          os.path.join(sample_dir, 'real_images.png'))
            
            fake_images_save = fake_images.reshape(fake_images.size(0), 1, 28, 28)
            save_image(denorm(fake_images_save[:64]),
                      os.path.join(sample_dir, f'fake_images_epoch_{epoch+1}.png'))
    
    # 训练完成后进行评估
    return _evaluate_experiment(exp_name, config, G, data_loader, device, sample_dir, description, results)

def _evaluate_experiment(exp_name, config, G, D, data_loader, device, sample_dir, description, results=None):
    """评估实验的公共函数"""
    if results is None:
        results = {}
    
    # 绘制loss曲线
    if results:
        save_loss_curves(results, exp_name, sample_dir)
    
    # 计算FID - 从exp_config获取正确的配置
    logging.info("Calculating FID score...")
    with torch.no_grad():
        real_images = []
        # 直接从数据加载器获取真实图像
        for img_batch, _ in data_loader:
            batch_size = img_batch.size(0)
            real_images.append(img_batch.to(device))
            if len(real_images) * batch_size >= 1000:
                break
        real_images = torch.cat(real_images, dim=0)[:1000]
        
        # 从exp_config获取正确的latent_size
        from exp_config import experiments, base_config
        if exp_name in experiments:
            exp_config = base_config.copy()
            exp_config.update(experiments[exp_name])
            latent_size = exp_config['latent_size']
        else:
            latent_size = base_config['latent_size']
        
        logging.info(f"Using latent_size: {latent_size} for FID calculation")
        
        z = torch.randn(1000, latent_size).to(device)
        fake_images = G(z).reshape(-1, 1, 28, 28)
        fid_score = calculate_fid(real_images, fake_images, device)
    
    # latent space analysis
    logging.info("Analyzing latent space...")
    with torch.no_grad():
        images_normal, images_far = analyze_latent_space(G, device, latent_size, num_samples=64)
        save_image(denorm(images_normal[:64]), os.path.join(sample_dir, 'latent_normal.png'))
        save_image(denorm(images_far[:64]), os.path.join(sample_dir, 'latent_far.png'))

    # 保存最终模型 - 生成器和判别器都要保存
    torch.save(G.state_dict(), os.path.join(MODEL_DIR, f'{exp_name}_G_final.pth'))
    torch.save(D.state_dict(), os.path.join(MODEL_DIR, f'{exp_name}_D_final.pth'))
    logging.info(f"Saved final models to {MODEL_DIR}")
    
    # 整理结果
    exp_results = {
        'description': description,
        'config': config,
        'fid_score': fid_score,
        'final_d_loss': results.get('epoch_d_losses', [0])[-1] if results.get('epoch_d_losses') else 0,
        'final_g_loss': results.get('epoch_g_losses', [0])[-1] if results.get('epoch_g_losses') else 0,
        'loss_curves': {
            'epoch_d_losses': results.get('epoch_d_losses', []),
            'epoch_g_losses': results.get('epoch_g_losses', [])
        },
        'model_paths': {
            'generator': os.path.join(MODEL_DIR, f'{exp_name}_G_final.pth'),
            'discriminator': os.path.join(MODEL_DIR, f'{exp_name}_D_final.pth')
        }
    }
    
    logging.info(f"Experiment {exp_name} completed. FID: {fid_score:.2f}")
    return exp_results


def main():
    """运行所有实验"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GAN Training with Reload Support')
    parser.add_argument('--reload', action='store_true', help='Reload from latest checkpoint if available')
    args = parser.parse_args()
    
    all_results = {}
    
    # 确保所有必要的目录都存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    for exp_name, config in experiments.items():
        try:
            results = train_experiment(exp_name, config, reload=args.reload)
            all_results[exp_name] = results
        except Exception as e:
            logging.error(f"Experiment {exp_name} failed: {str(e)}")
            continue
    
    # 保存结果
    with open('all_experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 输出总结
    logging.info("\n=== EXPERIMENT SUMMARY ===")
    for exp_name, results in all_results.items():
        logging.info(f"{exp_name:25} | FID: {results['fid_score']:6.2f} | "
                    f"Final D-Loss: {results['final_d_loss']:.4f} | "
                    f"G-Loss: {results['final_g_loss']:.4f} | "
                    f"Desc: {results['description'][:20]}...")
    
    logging.info(f"\nAll models saved to: {MODEL_DIR}")
    logging.info("All experiments completed!")

if __name__ == "__main__":
    main()
