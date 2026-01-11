# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import transformers
from transformers import AutoModel, BertTokenizerFast, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
import time
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os
from tqdm.auto import tqdm
import warnings
from exp_config import ExperimentConfig, EXPERIMENTS

warnings.filterwarnings('ignore')

# 设置matplotlib样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 创建输出目录
output_dir = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 重定向输出到文件
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(os.path.join(output_dir, "experiment_log.txt"))

# BERT模型架构
class BERT_Sentiment(nn.Module):
    def __init__(self, bert, dropout_rate=0.1, use_cls_token=True, token_position="cls"):
        super(BERT_Sentiment, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        # 对于first_last拼接，输入维度是1536
        if token_position == "first_last":
            fc1_input_dim = 1536
        else:
            fc1_input_dim = 768
            
        self.fc1 = nn.Linear(fc1_input_dim, 512) 
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.use_cls_token = use_cls_token
        self.token_position = token_position

    def freeze_bert_layers(self, num_layers_to_freeze):
        """冻结BERT的前n层"""
        if num_layers_to_freeze == -1:  # 冻结全部
            for param in self.bert.parameters():
                param.requires_grad = False
            print(f"Froze all BERT layers")
        elif num_layers_to_freeze > 0:  # 冻结前n层
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < num_layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"Froze first {num_layers_to_freeze} BERT layers")
        else:  # 不冻结
            print("No BERT layers frozen")

    def forward(self, sent_id, mask):
        outputs = self.bert(sent_id, attention_mask=mask)
        
        if self.use_cls_token:
            x = outputs.pooler_output
        else:
            last_hidden_state = outputs.last_hidden_state
            
            if self.token_position == "mean":
                input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                x = sum_embeddings / sum_mask
            elif self.token_position == "max":
                input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                last_hidden_state[input_mask_expanded == 0] = -1e9
                x = torch.max(last_hidden_state, 1)[0]
            elif self.token_position == "first_last":
                # 第一个token和最后一个token的拼接
                first_token = last_hidden_state[:, 0, :]  # [CLS] token
                last_token = last_hidden_state[:, -1, :]  # Last token
                x = torch.cat([first_token, last_token], dim=1)
            else:
                x = outputs.pooler_output
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x

# 训练和评估函数
def format_time(elapsed):
    return str(int((elapsed) / 60)) + ':' + str(int((elapsed) % 60))

def train_model(model, train_dataloader, optimizer, cross_entropy, device, scheduler=None, gradient_accumulation=False):
    model.train()
    total_loss = 0
    total_preds = []
    
    accumulation_steps = 4 if gradient_accumulation else 1
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_dataloader, desc="Training", leave=False,miniters = 100, disable=True)
    for step, batch in enumerate(progress_bar):
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        
        # 梯度累积
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'avg_loss': f'{total_loss/(step+1):.4f}'
        })
    
    # 处理剩余的梯度
    if gradient_accumulation and len(train_dataloader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

def evaluate_model(model, dataloader, cross_entropy, device):
    model.eval()
    total_loss = 0
    total_preds = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False,disable=True)
    for step, batch in enumerate(progress_bar):
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
            
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(step+1):.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# 加载IMDB数据集
print("Loading IMDB dataset...")
try:
    df = pd.read_csv("IMDB Dataset.csv")
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # 随机打乱数据
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print("Dataset shuffled successfully")
        
except Exception as e:
    print(f"Error loading IMDB Dataset.csv: {e}")
    print("Please make sure IMDB Dataset.csv is in the current directory")
    exit(1)

# 转换标签
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("Dataset Info:")
print(f"Total samples: {df.shape[0]}")
print(f"Positive reviews: {(df['label'] == 1).sum()}")
print(f"Negative reviews: {(df['label'] == 0).sum()}")

# 预加载BERT模型和tokenizer
print("Preloading BERT model and tokenizer...")
bert_model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = bert_model.to(device)
print("BERT model and tokenizer loaded successfully.")

# 预处理数据
print("Preprocessing data...")
# 一次性分割数据
train_text, temp_text, train_labels, temp_labels = train_test_split(
    df['review'], df['label'], 
    random_state=seed,
    train_size=0.7,
    stratify=df['label']
)

val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels,
    random_state=seed,
    test_size=0.5,
    stratify=temp_labels
)

print(f"Training samples: {len(train_text)}")
print(f"Validation samples: {len(val_text)}")
print(f"Test samples: {len(test_text)}")

# 一次性tokenization函数
def tokenize_dataset(texts, max_seq_len=128):
    return tokenizer.batch_encode_plus(
        texts.tolist(),
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )

# 为不同序列长度预先tokenize
max_seq_lens = [128]  # 只使用128长度加速实验
tokenized_data = {}

for seq_len in max_seq_lens:
    print(f"Tokenizing with max_seq_len={seq_len}...")
    tokenized_data[seq_len] = {
        'train': tokenize_dataset(train_text, seq_len),
        'val': tokenize_dataset(val_text, seq_len),
        'test': tokenize_dataset(test_text, seq_len)
    }

print("Data preprocessing completed.")

# 实验运行函数
def run_experiment(config, tokenized_data):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {config.exp_id}: {config.description}")
    print(f"{'='*60}")
    print(f"Config: train_size={config.train_size}, freeze_layers={config.freeze_layers}")
    print(f"         lr={config.learning_rate}, batch_size={config.batch_size}")
    print(f"         dropout={config.dropout_rate}, epochs={config.epochs}")
    print(f"         use_cls_token={config.use_cls_token}, token_position={config.token_position}")
    print(f"         class_weights={config.use_class_weights}, scheduler={config.use_scheduler}")
    
    # 获取预处理的tokenized数据
    tokens = tokenized_data[config.max_seq_len]
    
    # 根据训练大小调整训练数据
    if config.train_size < 0.7:
        adjusted_train_size = int(len(train_text) * config.train_size / 0.7)
        train_indices = np.random.choice(len(train_text), adjusted_train_size, replace=False)
        
        train_seq = torch.tensor(np.array(tokens['train']['input_ids'])[train_indices])
        train_mask = torch.tensor(np.array(tokens['train']['attention_mask'])[train_indices])
        train_y = torch.tensor(train_labels.iloc[train_indices].tolist())
    else:
        train_seq = torch.tensor(tokens['train']['input_ids'])
        train_mask = torch.tensor(tokens['train']['attention_mask'])
        train_y = torch.tensor(train_labels.tolist())
    
    val_seq = torch.tensor(tokens['val']['input_ids'])
    val_mask = torch.tensor(tokens['val']['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())
    
    test_seq = torch.tensor(tokens['test']['input_ids'])
    test_mask = torch.tensor(tokens['test']['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())
    
    # 创建DataLoaders
    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)
    
    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=config.batch_size)
    
    test_data = TensorDataset(test_seq, test_mask, test_y)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config.batch_size)
    
    # 复制BERT模型
    bert_copy = AutoModel.from_pretrained('bert-base-uncased')
    
    # 初始化模型
    model = BERT_Sentiment(
        bert_copy, 
        dropout_rate=config.dropout_rate,
        use_cls_token=config.use_cls_token,
        token_position=config.token_position
    )
    
    # 应用冻结策略
    if config.freeze_layers is not None:
        model.freeze_bert_layers(config.freeze_layers)
    
    model = model.to(device)
    
    # 优化器 - 只优化需要梯度的参数
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(optimizer_params, lr=config.learning_rate)
    
    # 学习率调度器
    total_steps = len(train_dataloader) * config.epochs
    if config.use_scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
    else:
        scheduler = None
    
    # 损失函数 - 添加类别权重
    if config.use_class_weights:
        class_wts = compute_class_weight(
            'balanced', 
            classes=np.unique(train_labels), 
            y=train_labels
        )
        weights = torch.tensor(class_wts, dtype=torch.float).to(device)
        cross_entropy = nn.NLLLoss(weight=weights)
        print(f"Using class weights: {class_wts}")
    else:
        cross_entropy = nn.NLLLoss()
    
    # 训练循环
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    
    print(f"Starting training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        print(f'\nEpoch {epoch + 1}/{config.epochs}')
        print('-' * 50)
        
        # 训练
        train_loss, train_preds = train_model(
            model, train_dataloader, optimizer, cross_entropy, device, 
            scheduler, config.gradient_accumulation
        )
        train_preds = np.argmax(train_preds, axis=1)
        train_acc = accuracy_score(train_y.numpy(), train_preds)
        
        # 验证
        valid_loss, valid_preds = evaluate_model(model, val_dataloader, cross_entropy, device)
        valid_preds = np.argmax(valid_preds, axis=1)
        valid_acc = accuracy_score(val_y.numpy(), valid_preds)
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f'saved_weights_exp_{config.exp_id}.pt'))
        
        # 记录指标
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)
        
        print(f'Training Loss: {train_loss:.3f} | Training Accuracy: {train_acc:.3f}')
        print(f'Validation Loss: {valid_loss:.3f} | Validation Accuracy: {valid_acc:.3f}')
        print(f'Best Validation Loss: {best_valid_loss:.3f}')
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(os.path.join(output_dir, f'saved_weights_exp_{config.exp_id}.pt')))
    
    # 测试评估
    test_loss, test_preds = evaluate_model(model, test_dataloader, cross_entropy, device)
    test_preds_class = np.argmax(test_preds, axis=1)
    
    accuracy = accuracy_score(test_y, test_preds_class)
    f1 = f1_score(test_y, test_preds_class, average='weighted')
    
    print(f"\nTest Results - Loss: {test_loss:.3f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 返回结果
    results = {
        'exp_id': config.exp_id,
        'train_size': config.train_size,
        'freeze_layers': config.freeze_layers,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'dropout_rate': config.dropout_rate,
        'epochs': config.epochs,
        'max_seq_len': config.max_seq_len,
        'use_cls_token': config.use_cls_token,
        'token_position': config.token_position,
        'use_class_weights': config.use_class_weights,
        'use_scheduler': config.use_scheduler,
        'gradient_accumulation': config.gradient_accumulation,
        'description': config.description,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accuracies': train_accuracies,
        'valid_accuracies': valid_accuracies,
        'test_loss': test_loss,
        'test_accuracy': accuracy,
        'test_f1': f1,
        'test_predictions': test_preds_class,
        'test_true_labels': test_y.numpy()
    }
    
    return results

# 运行所有实验
print("Starting all experiments...")
all_results = []

for config in EXPERIMENTS:
    print(f"\n>>> Starting Experiment {config.exp_id}")
    try:
        results = run_experiment(config, tokenized_data)
        all_results.append(results)
        print(f">>> ✓ Experiment {config.exp_id} completed successfully.")
    except Exception as e:
        print(f">>> ✗ Experiment {config.exp_id} failed: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\nCompleted {len(all_results)} out of {len(EXPERIMENTS)} experiments.")

# 保存结果到CSV
print("\nSaving results to CSV...")

results_data = []
for result in all_results:
    results_data.append({
        'exp_id': result['exp_id'],
        'description': result['description'],
        'train_size': result['train_size'],
        'freeze_layers': result['freeze_layers'],
        'learning_rate': result['learning_rate'],
        'batch_size': result['batch_size'],
        'dropout_rate': result['dropout_rate'],
        'epochs': result['epochs'],
        'max_seq_len': result['max_seq_len'],
        'use_cls_token': result['use_cls_token'],
        'token_position': result['token_position'],
        'use_class_weights': result['use_class_weights'],
        'use_scheduler': result['use_scheduler'],
        'gradient_accumulation': result['gradient_accumulation'],
        'test_loss': result['test_loss'],
        'test_accuracy': result['test_accuracy'],
        'test_f1': result['test_f1'],
        'final_train_loss': result['train_losses'][-1] if result['train_losses'] else None,
        'final_val_loss': result['valid_losses'][-1] if result['valid_losses'] else None,
        'final_train_acc': result['train_accuracies'][-1] if result['train_accuracies'] else None,
        'final_val_acc': result['valid_accuracies'][-1] if result['valid_accuracies'] else None,
    })

results_df = pd.DataFrame(results_data)
csv_path = os.path.join(output_dir, "experiment_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"Results saved to: {csv_path}")

# 保存详细训练历史
detailed_results = []
for result in all_results:
    for epoch in range(len(result['train_losses'])):
        detailed_results.append({
            'exp_id': result['exp_id'],
            'epoch': epoch + 1,
            'train_loss': result['train_losses'][epoch],
            'val_loss': result['valid_losses'][epoch],
            'train_acc': result['train_accuracies'][epoch],
            'val_acc': result['valid_accuracies'][epoch]
        })

detailed_df = pd.DataFrame(detailed_results)
detailed_csv_path = os.path.join(output_dir, "training_history.csv")
detailed_df.to_csv(detailed_csv_path, index=False)
print(f"Detailed training history saved to: {detailed_csv_path}")

# 显示汇总结果
print("\nExperiment Results Summary:")
print(results_df[['exp_id', 'description', 'test_accuracy', 'test_f1', 'test_loss']].to_string(index=False))

# 创建可视化
print("\nCreating visualizations...")

# 设置绘图风格
plt.style.use('default')
sns.set_palette("deep")

# 1. 准确率对比图
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
exp_ids = [f"Exp {r['exp_id']}" for r in all_results]
accuracies = [r['test_accuracy'] for r in all_results]

# 使用更深的颜色，加粗边框
colors = ['#1f77b4' if acc != max(accuracies) else '#d62728' for acc in accuracies]

bars = plt.bar(exp_ids, accuracies, color=colors, alpha=0.9, 
               edgecolor='black', linewidth=2)

plt.title('Test Accuracy by Experiment', fontsize=16, fontweight='bold')
plt.xlabel('Experiment ID', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 在柱子上添加数值
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', 
             fontweight='bold', fontsize=9)

# 2. 超参数影响分析
plt.subplot(1, 2, 2)
# 创建token策略列
token_strategies = []
for result in all_results:
    if result['use_cls_token']:
        token_strategies.append('CLS')
    else:
        token_strategies.append(result['token_position'].upper())

strategy_df = pd.DataFrame({
    'strategy': token_strategies,
    'accuracy': accuracies
})

strategy_means = strategy_df.groupby('strategy')['accuracy'].mean()

bars2 = plt.bar(strategy_means.index, strategy_means.values, alpha=0.9, 
                edgecolor='black', linewidth=2, color=['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b'])
plt.title('Average Accuracy by Token Strategy', fontsize=16, fontweight='bold')
plt.xlabel('Token Strategy', fontsize=12, fontweight='bold')
plt.ylabel('Average Accuracy', fontsize=12, fontweight='bold')
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, mean in zip(bars2, strategy_means.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. 训练曲线示例
if len(all_results) >= 4:
    plt.figure(figsize=(15, 10))
    selected_exps = [0, 4, 8, 12]  # 选择有代表性的实验
    
    for i, exp_idx in enumerate(selected_exps):
        if exp_idx < len(all_results):
            result = all_results[exp_idx]
            
            plt.subplot(2, 2, i+1)
            plt.plot(result['train_losses'], label='Training Loss', linewidth=2, marker='o', markersize=3)
            plt.plot(result['valid_losses'], label='Validation Loss', linewidth=2, marker='s', markersize=3)
            plt.title(f'Exp {result["exp_id"]}: {result["description"][:30]}...\n(Test Acc: {result["test_accuracy"]:.3f})', 
                     fontweight='bold', fontsize=10)
            plt.xlabel('Epoch', fontweight='bold')
            plt.ylabel('Loss', fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()

print(f"\nAll results and visualizations saved to: {output_dir}")
print("Experiment completed successfully!")

# 保存实验配置
config_df = pd.DataFrame([{
    'total_experiments': len(EXPERIMENTS),
    'completed_experiments': len(all_results),
    'device_used': str(device),
    'dataset_size': len(df),
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}])
config_df.to_csv(os.path.join(output_dir, 'experiment_config.csv'), index=False)