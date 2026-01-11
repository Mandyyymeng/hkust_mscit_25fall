# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import transformers
from transformers import AutoModel, BertTokenizerFast
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
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

# 设置设备 - 优先使用GPU
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


"""# Load and Prepare IMDB Dataset"""

print("Loading IMDB dataset...")
try:
    # 尝试加载真实IMDB数据集
    df = pd.read_csv("IMDB Dataset.csv")
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # 确保数据多样性 - 随机打乱
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 如果数据太大，可以采样一部分用于快速实验
    if len(df) > 10000:
        print("Sampling 10,000 examples for faster experimentation...")
        df = df.sample(n=10000, random_state=seed).reset_index(drop=True)
        
except Exception as e:
    print(f"Error loading IMDB Dataset.csv: {e}")
    print("Creating diverse sample data for testing...")
    # 创建更多样化的模拟数据
    base_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the storyline captivating.",
        "I hated this film. It was boring, poorly made, and a complete waste of time.",
        "The movie was okay, nothing special but not terrible either. Average entertainment.",
        "An incredible masterpiece that deserves all the awards! Brilliant direction and acting.",
        "Waste of time. Poor storyline, bad acting, and terrible cinematography.",
        "Brilliant cinematography and excellent performances throughout the entire film.",
        "Disappointing sequel that fails to capture the magic of the original movie.",
        "One of the best movies I've seen this year, highly recommended to everyone!",
        "Terrible plot with wooden acting. Would not watch again under any circumstances.",
        "A heartwarming story with great character development and emotional depth.",
        "Visually stunning but the plot was confusing and hard to follow at times.",
        "Outstanding performance by the lead actor, but the supporting cast was weak.",
        "The special effects were impressive, but the story lacked originality.",
        "A perfect blend of comedy and drama that kept me engaged from start to finish.",
        "Poorly written dialogue and unconvincing characters made this film unbearable.",
        "The director's vision was clear and executed beautifully in this cinematic gem.",
        "Despite the hype, this movie failed to deliver on its promises and fell flat.",
        "A thought-provoking film that raises important questions about society today.",
        "The pacing was too slow and the runtime could have been shortened significantly.",
        "An unforgettable experience that will stay with me for a long time to come."
    ]
    
    # 生成更多样化的数据
    sample_reviews = []
    sample_sentiments = []
    
    for i in range(5000):  # 5000个样本
        base_idx = i % len(base_reviews)
        review = base_reviews[base_idx]
        # 添加一些变化
        if i > len(base_reviews):
            review = review.replace("movie", random.choice(["film", "picture", "feature"]))
            review = review.replace("good", random.choice(["great", "excellent", "wonderful"]))
            review = review.replace("bad", random.choice(["terrible", "awful", "horrible"]))
        
        sample_reviews.append(review)
        # 基于内容分配情感标签
        positive_words = ['fantastic', 'superb', 'incredible', 'brilliant', 'excellent', 
                         'best', 'heartwarming', 'outstanding', 'perfect', 'beautifully',
                         'thought-provoking', 'unforgettable', 'great', 'wonderful']
        negative_words = ['hated', 'boring', 'poorly', 'waste', 'terrible', 'disappointing',
                         'failed', 'poor', 'unbearable', 'fell flat', 'awful', 'horrible']
        
        if any(word in review.lower() for word in positive_words):
            sample_sentiments.append('positive')
        elif any(word in review.lower() for word in negative_words):
            sample_sentiments.append('negative')
        else:
            sample_sentiments.append(random.choice(['positive', 'negative']))
    
    df = pd.DataFrame({
        'review': sample_reviews,
        'sentiment': sample_sentiments
    })

# 转换标签
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("Dataset Info:")
print(f"Total samples: {df.shape[0]}")
print(f"Positive reviews: {(df['label'] == 1).sum()}")
print(f"Negative reviews: {(df['label'] == 0).sum()}")

"""# Experiment Configuration Class"""

class ExperimentConfig:
    def __init__(self, 
                 exp_id=1,
                 train_size=0.7, 
                 freeze_bert=True,
                 learning_rate=1e-5,
                 batch_size=32,
                 dropout_rate=0.1,
                 epochs=5,
                 max_seq_len=128,  # 减少序列长度加速训练
                 use_cls_token=True,
                 token_position="cls",
                 description=""
                 ):
        self.exp_id = exp_id
        self.train_size = train_size
        self.freeze_bert = freeze_bert
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.max_seq_len = max_seq_len
        self.use_cls_token = use_cls_token
        self.token_position = token_position
        self.description = description

"""# BERT Model Architecture"""

class BERT_Sentiment(nn.Module):
    def __init__(self, bert, dropout_rate=0.1, use_cls_token=True, token_position="cls"):
        super(BERT_Sentiment, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout_rate)
        # relu activation function
        self.relu =  nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2) # HAM vs SPAM (2 LABELS)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)
        self.use_cls_token = use_cls_token
        self.token_position = token_position

    def forward(self, sent_id, mask):
        outputs = self.bert(sent_id, attention_mask=mask)
        # added
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
            else:
                x = outputs.pooler_output
        
        # colab
        x = self.fc1(x)  # (outputs.pooler_output)
        #x dim 512
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)
        
        return x

"""# Training and Evaluation Functions"""

def format_time(elapsed):
    return str(int((elapsed) / 60)) + ':' + str(int((elapsed) % 60))

def train_model(model, train_dataloader, optimizer, cross_entropy, device):
    model.train()
    total_loss = 0
    total_preds = []
    
    progress_bar = tqdm(train_dataloader, desc="Training", leave=False,miniters=100,disable=True)
    for step, batch in enumerate(progress_bar):
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        
        model.zero_grad()
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss += loss.item()
        
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()
        # append the model predictions
        total_preds.append(preds)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(step+1):.4f}'
        })
    
    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


def evaluate_model(model, dataloader, cross_entropy, device):
    model.eval()
    total_loss = 0
    total_preds = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False,miniters=50,disable=True)
    for step, batch in enumerate(progress_bar):
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch  ### todo: to_device() ? 
        
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

"""# Preload BERT Model and Tokenizer (一次性加载)"""

# from colab: 
print("Preloading BERT model and tokenizer...")
bert_model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# 将模型移到设备
bert_model = bert_model.to(device)
print("BERT model and tokenizer loaded successfully.")

"""# Preprocess Data (一次性预处理)"""

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
max_seq_lens = [64, 128, 256]
tokenized_data = {}

for seq_len in max_seq_lens:
    print(f"Tokenizing with max_seq_len={seq_len}...")
    tokenized_data[seq_len] = {
        'train': tokenize_dataset(train_text, seq_len),
        'val': tokenize_dataset(val_text, seq_len),
        'test': tokenize_dataset(test_text, seq_len)
    }

print("Data preprocessing completed.")

"""# Experiment Runner"""

def run_experiment(config, tokenized_data):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {config.exp_id}: {config.description}")
    print(f"{'='*60}")
    print(f"Config: train_size={config.train_size}, freeze_bert={config.freeze_bert}")
    print(f"         lr={config.learning_rate}, batch_size={config.batch_size}")
    print(f"         epochs={config.epochs}, seq_len={config.max_seq_len}")
    print(f"         use_cls_token={config.use_cls_token}, token_position={config.token_position}")
    
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
        # colab
        train_seq = torch.tensor(tokens['train']['input_ids'])
        train_mask = torch.tensor(tokens['train']['attention_mask'])
        train_y = torch.tensor(train_labels.tolist())
    
    # colab: integer seq to tensor
    val_seq = torch.tensor(tokens['val']['input_ids'])
    val_mask = torch.tensor(tokens['val']['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())
    
    test_seq = torch.tensor(tokens['test']['input_ids'])
    test_mask = torch.tensor(tokens['test']['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())
    
    # colab: create DataLoaders
    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)
    
    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=config.batch_size)
    
    test_data = TensorDataset(test_seq, test_mask, test_y)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config.batch_size)
    
    # 复制BERT模型（而不是重新加载）
    bert_copy = AutoModel.from_pretrained('bert-base-uncased')
    if config.freeze_bert:
        # colab
        for param in bert_copy.parameters():
            param.requires_grad = False
    
    # 初始化模型
    model = BERT_Sentiment(
        bert_copy, 
        dropout_rate=config.dropout_rate,
        use_cls_token=config.use_cls_token,
        token_position=config.token_position
    )
    model = model.to(device)
    
    # 优化器 - 只优化需要梯度的参数
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(optimizer_params, lr=config.learning_rate)
    
    # 损失函数
    cross_entropy = nn.NLLLoss()
    
    # 训练循环
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    
    print(f"Starting training for {config.epochs} epochs...")
    
    # 使用tqdm进度条
    epoch_progress = tqdm(range(config.epochs), desc="Epochs",miniters=100)
    
    for epoch in epoch_progress:
        epoch_progress.set_description(f"Epoch {epoch + 1}/{config.epochs}")
        
        # 训练
        train_loss, train_preds = train_model(model, train_dataloader, optimizer, cross_entropy, device)
        train_preds = np.argmax(train_preds, axis=1)
        train_acc = accuracy_score(train_y.numpy(), train_preds) # colab: not computing acc, only loss
        
        # 验证
        valid_loss, valid_preds = evaluate_model(model, val_dataloader, cross_entropy, device)
        valid_preds = np.argmax(valid_preds, axis=1)
        valid_acc = accuracy_score(val_y.numpy(), valid_preds)
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f'saved_weights_exp_{config.exp_id}.pt'))
            # colab: torch.save(model.state_dict(), 'saved_weights.pt')
        
        # 记录指标
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)
        
        # 更新进度条
        epoch_progress.set_postfix({
            'train_loss': f'{train_loss:.3f}',
            'val_loss': f'{valid_loss:.3f}',
            'train_acc': f'{train_acc:.3f}',
            'val_acc': f'{valid_acc:.3f}',
            'best_val': f'{best_valid_loss:.3f}'
        })
    
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
        'freeze_bert': config.freeze_bert,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'dropout_rate': config.dropout_rate,
        'epochs': config.epochs,
        'max_seq_len': config.max_seq_len,
        'use_cls_token': config.use_cls_token,
        'token_position': config.token_position,
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

"""# Define and Run Experiments"""

# 定义10个有意义的实验
experiments = [
    ExperimentConfig(exp_id=1, train_size=0.3, freeze_bert=True, learning_rate=1e-5, 
                    batch_size=16, epochs=3, max_seq_len=128, use_cls_token=True,
                    description="Baseline: 30% data, frozen BERT, CLS token"),
    
    ExperimentConfig(exp_id=2, train_size=0.7, freeze_bert=True, learning_rate=1e-5, 
                    batch_size=32, epochs=5, max_seq_len=128, use_cls_token=True,
                    description="More data: 70% data, frozen BERT"),
    
    ExperimentConfig(exp_id=3, train_size=0.7, freeze_bert=False, learning_rate=2e-5, 
                    batch_size=16, epochs=3, max_seq_len=128, use_cls_token=True,
                    description="Full fine-tuning: unfrozen BERT, higher LR"),
    
    ExperimentConfig(exp_id=4, train_size=0.7, freeze_bert=True, learning_rate=5e-5, 
                    batch_size=32, epochs=5, max_seq_len=128, use_cls_token=True,
                    description="Higher learning rate"),
    
    ExperimentConfig(exp_id=5, train_size=0.7, freeze_bert=True, learning_rate=1e-5, 
                    batch_size=64, epochs=5, max_seq_len=128, use_cls_token=True,
                    description="Larger batch size"),
    
    ExperimentConfig(exp_id=6, train_size=0.7, freeze_bert=True, learning_rate=1e-5, 
                    batch_size=32, dropout_rate=0.3, epochs=5, max_seq_len=128, use_cls_token=True,
                    description="Higher dropout for regularization"),
    
    ExperimentConfig(exp_id=7, train_size=0.7, freeze_bert=True, learning_rate=1e-5, 
                    batch_size=32, epochs=8, max_seq_len=128, use_cls_token=True,
                    description="More training epochs"),
    
    ExperimentConfig(exp_id=8, train_size=0.7, freeze_bert=True, learning_rate=1e-5, 
                    batch_size=32, epochs=5, max_seq_len=128, use_cls_token=False, token_position="mean",
                    description="Mean pooling instead of CLS token"),
    
    ExperimentConfig(exp_id=9, train_size=0.7, freeze_bert=True, learning_rate=1e-5, 
                    batch_size=32, epochs=5, max_seq_len=128, use_cls_token=False, token_position="max",
                    description="Max pooling instead of CLS token"),
    
    ExperimentConfig(exp_id=10, train_size=0.8, freeze_bert=False, learning_rate=2e-5, 
                    batch_size=32, dropout_rate=0.2, epochs=5, max_seq_len=128, use_cls_token=True,
                    description="Best combination: more data, unfrozen, tuned params"),
]

# 运行所有实验
print("Starting all experiments...")
all_results = []

# 使用tqdm显示总体进度
exp_progress = tqdm(experiments, desc="Experiments",miniters=50,disable=True) #disable=True
for exp_config in exp_progress:
    exp_progress.set_description(f"Experiment {exp_config.exp_id}")
    try:
        results = run_experiment(exp_config, tokenized_data)
        all_results.append(results)
        print(f"✓ Experiment {exp_config.exp_id} completed successfully.")
    except Exception as e:
        print(f"✗ Experiment {exp_config.exp_id} failed: {e}")
        continue

print(f"\nCompleted {len(all_results)} out of {len(experiments)} experiments.")

"""# Save Results to CSV"""

print("\nSaving results to CSV...")

# 创建结果DataFrame
results_data = []
for result in all_results:
    results_data.append({
        'exp_id': result['exp_id'],
        'description': result['description'],
        'train_size': result['train_size'],
        'freeze_bert': result['freeze_bert'],
        'learning_rate': result['learning_rate'],
        'batch_size': result['batch_size'],
        'dropout_rate': result['dropout_rate'],
        'epochs': result['epochs'],
        'max_seq_len': result['max_seq_len'],
        'use_cls_token': result['use_cls_token'],
        'token_position': result['token_position'],
        'test_loss': result['test_loss'],
        'test_accuracy': result['test_accuracy'],
        'test_f1': result['test_f1'],
        'final_train_loss': result['train_losses'][-1] if result['train_losses'] else None,
        'final_val_loss': result['valid_losses'][-1] if result['valid_losses'] else None,
        'final_train_acc': result['train_accuracies'][-1] if result['train_accuracies'] else None,
        'final_val_acc': result['valid_accuracies'][-1] if result['valid_accuracies'] else None,
    })

results_df = pd.DataFrame(results_data)

# 保存到CSV
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

"""# Create Comprehensive Visualizations"""

print("\nCreating visualizations...")

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. 准确率对比图
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
exp_ids = [f"Exp {r['exp_id']}" for r in all_results]
accuracies = [r['test_accuracy'] for r in all_results]
colors = ['lightblue' if acc != max(accuracies) else 'red' for acc in accuracies]

bars = plt.bar(exp_ids, accuracies, color=colors, alpha=0.7, edgecolor='black')
plt.title('Test Accuracy by Experiment', fontsize=14, fontweight='bold')
plt.xlabel('Experiment ID')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# 在柱子上添加数值
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

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

plt.bar(strategy_means.index, strategy_means.values, alpha=0.7, edgecolor='black')
plt.title('Average Accuracy by Token Strategy', fontsize=14, fontweight='bold')
plt.xlabel('Token Strategy')
plt.ylabel('Average Accuracy')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. 训练损失和准确率曲线
plt.figure(figsize=(15, 10))

# 选择几个有代表性的实验
selected_exps = [0, 2, 6, min(range(len(all_results)), key=lambda i: all_results[i]['test_loss'])]

for i, exp_idx in enumerate(selected_exps):
    if exp_idx < len(all_results):
        result = all_results[exp_idx]
        
        # 损失曲线
        plt.subplot(2, 2, i+1)
        plt.plot(result['train_losses'], label='Training Loss', linewidth=2, marker='o', markersize=3)
        plt.plot(result['valid_losses'], label='Validation Loss', linewidth=2, marker='s', markersize=3)
        plt.title(f'Exp {result["exp_id"]}: Loss Curves\n(Test Acc: {result["test_accuracy"]:.3f})', 
                 fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. 准确率曲线
plt.figure(figsize=(15, 10))

for i, exp_idx in enumerate(selected_exps):
    if exp_idx < len(all_results):
        result = all_results[exp_idx]
        
        plt.subplot(2, 2, i+1)
        plt.plot(result['train_accuracies'], label='Training Accuracy', linewidth=2, marker='o', markersize=3)
        plt.plot(result['valid_accuracies'], label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
        plt.title(f'Exp {result["exp_id"]}: Accuracy Curves\n(Test Acc: {result["test_accuracy"]:.3f})', 
                 fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
plt.show()

# 5. 超参数热力图
plt.figure(figsize=(12, 8))
# 选择数值型参数进行热力图分析
numeric_params = ['train_size', 'learning_rate', 'batch_size', 'dropout_rate', 'epochs', 'max_seq_len']
corr_data = results_df[['test_accuracy'] + numeric_params]

# 计算相关性矩阵
correlation_matrix = corr_data.corr()

# 绘制热力图
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={"shrink": .8},
            annot_kws={"size": 10})

plt.title('Hyperparameter Correlation with Test Accuracy', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()

# 6. 冻结vs不冻结BERT比较
freeze_results = [r for r in all_results if r['freeze_bert']]
unfreeze_results = [r for r in all_results if not r['freeze_bert']]

if freeze_results and unfreeze_results:
    plt.figure(figsize=(10, 6))
    
    freeze_acc = [r['test_accuracy'] for r in freeze_results]
    unfreeze_acc = [r['test_accuracy'] for r in unfreeze_results]
    
    categories = ['Frozen BERT', 'Unfrozen BERT']
    means = [np.mean(freeze_acc), np.mean(unfreeze_acc)]
    stds = [np.std(freeze_acc), np.std(unfreeze_acc)]
    
    bars = plt.bar(categories, means, yerr=stds, capsize=10, alpha=0.7, 
                   color=['lightblue', 'lightcoral'], edgecolor='black')
    
    plt.title('Performance: Frozen vs Unfrozen BERT', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy')
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'freeze_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

# 7. 训练数据量影响
plt.figure(figsize=(10, 6))
train_sizes = [r['train_size'] for r in all_results]
accuracies = [r['test_accuracy'] for r in all_results]

plt.scatter(train_sizes, accuracies, s=100, alpha=0.7, c=accuracies, cmap='viridis')
plt.colorbar(label='Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Test Accuracy')
plt.title('Effect of Training Data Size on Accuracy', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

# 添加趋势线
if len(train_sizes) > 1:
    z = np.polyfit(train_sizes, accuracies, 1)
    p = np.poly1d(z)
    plt.plot(train_sizes, p(train_sizes), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'data_size_effect.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAll results and visualizations saved to: {output_dir}")
print("Experiment completed successfully!")

# 保存实验配置
config_df = pd.DataFrame([{
    'total_experiments': len(experiments),
    'completed_experiments': len(all_results),
    'device_used': str(device),
    'dataset_size': len(df),
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}])
config_df.to_csv(os.path.join(output_dir, 'experiment_config.csv'), index=False)