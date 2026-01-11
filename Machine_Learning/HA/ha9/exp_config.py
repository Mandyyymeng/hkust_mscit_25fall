"""实验配置 - 包含GAN稳定技术和作业要求的实验"""

# 基础配置（更稳定的参数）
base_config = {
    'image_size': 784,           # 28x28
    'latent_size': 128,          # 隐变量维度
    'hidden_size': 256,          # 隐藏层大小
    'num_epochs': 10,            # 训练轮数 40
    'batch_size': 128,           # 批大小
    'lr': 1e-4,                # 学习率  0.0002
    'loss_function': 'bce',      # 损失函数
    'disc_leaky_slope': 0.2,     # LeakyReLU负斜率
    'use_gradient_penalty': False,  # 是否使用梯度惩罚
    'gp_weight': 10.0,           # 梯度惩罚权重
    'use_spectral_norm': False,   # 是否使用谱归一化
    'n_critic': 1,               # 判别器训练次数
}


base_config = {
    'image_size': 784,    
    'latent_size': 128,   
    'hidden_size': 256,     
    'num_epochs': 10,          
    'batch_size': 128,           
    'lr': 1e-4,               
    'loss_function': 'bce',    
    'disc_leaky_slope': 0.2,    
    'use_gradient_penalty': False,  
    'gp_weight': 10.0,          
    'use_spectral_norm': False,  
    'n_critic': 1,             
}

# 实验配置（15个实验）
experiments = { 
    # === 1. 基础实验 ===
    # 'exp14_spectral_norm': {
    #     'use_spectral_norm': True,
    #     'loss_function': 'wasserstein',
    #     'd_lr': 0.00005,      # 降低判别器学习率
    #     'g_lr': 0.0001,       # 生成器学习率稍高
    #     'n_critic': 3,        # 增加判别器训练次数
    #     'description': 'Stability: Spectral Norm + Wasserstein + conservative LR'
    # },
    'exp14_gradient_penalty_improved': {
        'use_gradient_penalty': True,
        'loss_function': 'wasserstein',
        'gp_weight': 10.0,
        'n_critic': 3,           # 减少critic次数
        'd_lr': 0.0001,          # 保守学习率
        'g_lr': 0.0001,
        'num_epochs': 20,        # 延长训练
        'description': 'WGAN-GP with balanced training'
    },
    # === 15-16. capacity ===
    'exp15_pure_capacity': {
        'hidden_size': 512,         # 单纯增加容量
        'latent_size': 128,         # 保持其他不变
        'num_epochs': 10,           # 保持训练轮数不变
        'lr': 1e-4,                 # 保持学习率不变
        'description': 'Pure Capacity: Only hidden_size↑ from 256→512, all else equal'
    },
    'exp16_capacity_adapted': {
        'hidden_size': 512,         # 增加容量
        'latent_size': 128,
        'num_epochs': 20,           # 延长训练适应更大容量
        'lr': 5e-5,                 # 降低学习率适应更大模型
        'd_lr': 4e-5,               # 更保守的判别器学习率
        'g_lr': 6e-5,               # 生成器学习率稍高
        'description': 'Capacity Adapted: hidden_size↑ + longer training + refined LR'
    },
    
    'exp1_baseline': {
        'description': 'Baseline: Standard GAN with BCE loss'
    },

    # === 5-6. 隐变量维度变化 (作业要求) ===
    'exp2_small_latent': {
        'latent_size': 64,
        'description': 'Latent: Smaller latent dimension (64)'
    },
    'exp3_large_latent': {
        'latent_size': 256,
        'description': 'Latent: Larger latent dimension (256)'
    },
    
    # === 7-8. 架构变化 - 激活函数 (作业要求) ===
    'exp4_elu_activation': {
        'activation': 'elu',
        'description': 'Architecture: ELU activation instead of ReLU/LeakyReLU'
    },
    'exp5_tanh_activation': {
        'activation': 'tanh',
        'description': 'Architecture: Tanh activation in hidden layers'
    },
    
    # === 9-10. 训练参数 - 学习率/轮数 (作业要求) ===
    'exp6_higher_lr': {
        'lr': 0.001,
        'description': 'Training: Higher learning rate (1e-3)'
    },
    'exp7_more_epochs': {
        'num_epochs': 80,
        'lr': 0.0001,
        'description': 'Training: More epochs (80) with lower LR'
    },
    
    # === 11-12. 生成器损失函数 (作业要求) ===
    'exp8_mse_loss': {
        'loss_function': 'mse',
        'description': 'Loss: MSE instead of BCE'
    },
    'exp9_wasserstein_loss': {
        'loss_function': 'wasserstein',
        'description': 'Loss: Wasserstein loss (remove sigmoid)'
    },
    
    # === 13-14. GAN稳定技术 ===
    'exp10_gradient_penalty': {
        'use_gradient_penalty': True,
        'gp_weight': 10.0,
        'n_critic': 5,
        'loss_function': 'wasserstein',
        'description': 'Stability: WGAN-GP with gradient penalty'
    },
    
    # === 11-13. 学习率对比实验 (G和D不同学习率) ===
    'exp11_equal_lr': {
        'd_lr': 0.0002,    # D学习率
        'g_lr': 0.0002,    # G学习率相同
        'description': 'LR: G_lr = D_lr (balanced)'
    },
    'exp12_g_larger_lr': {
        'd_lr': 0.0001,    # D学习率较小
        'g_lr': 0.0004,    # G学习率较大 (4倍)
        'description': 'LR: G_lr > D_lr (1:4 ratio)'
    },
    'exp13_g_smaller_lr': {
        'd_lr': 0.0004,    # D学习率较大
        'g_lr': 0.0001,    # G学习率较小 (1:4 ratio)
        'description': 'LR: G_lr < D_lr (1:4 ratio)'
    },
    
}
