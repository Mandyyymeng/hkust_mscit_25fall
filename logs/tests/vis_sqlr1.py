import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_data():
    """加载所有必要的数据"""
    # 读取diff.json
    sql_dir = "logs/pred_sqls/pred_sqls_r1"
    with open(f'{sql_dir}/birddev-generated_sql_eval_diff.json', 'r') as f:
        diff_data = json.load(f)
    
    # 读取dev.json
    with open('data/bird/dev/dev_all.json', 'r') as f:
        dev_data = json.load(f)
    
    # 创建question_id到difficulty的映射
    id_to_difficulty = {}
    for item in dev_data:
        id_to_difficulty[item["question_id"]] = item["difficulty"]
    
    return diff_data, id_to_difficulty

def calculate_database_accuracy(diff_data, id_to_difficulty):
    """计算每个数据库的准确率"""
    # 从dev.json中获取question_id到db_id的映射
    with open('data/bird/dev/dev_all.json', 'r') as f:
        dev_data = json.load(f)
    id_to_db = {item["question_id"]: item["db_id"] for item in dev_data}
    
    # 按数据库分组统计
    db_stats = defaultdict(lambda: {'total': 0, 'max_vote_correct': 0, 'upper_bound_correct': 0})
    
    for item in diff_data:
        question_id = item["question_id"]
        db_id = id_to_db.get(question_id)
        
        if db_id:
            db_stats[db_id]['total'] += 1
            if item["max_vote_correctness"] == 1:
                db_stats[db_id]['max_vote_correct'] += 1
            if item["upper_bound_correctness"] == 1:
                db_stats[db_id]['upper_bound_correct'] += 1
    
    # 计算准确率
    db_accuracy = {}
    for db_id, stats in db_stats.items():
        if stats['total'] > 0:
            db_accuracy[db_id] = {
                'max_vote_accuracy': stats['max_vote_correct'] / stats['total'],
                'upper_bound_accuracy': stats['upper_bound_correct'] / stats['total'],
                'total_questions': stats['total']
            }
    
    return db_accuracy

def calculate_difficulty_accuracy(diff_data, id_to_difficulty):
    """按难度分组计算准确率"""
    difficulty_stats = defaultdict(lambda: {'total': 0, 'max_vote_correct': 0, 'upper_bound_correct': 0})
    
    for item in diff_data:
        question_id = item["question_id"]
        difficulty = id_to_difficulty.get(question_id, 'unknown')
        
        difficulty_stats[difficulty]['total'] += 1
        if item["max_vote_correctness"] == 1:
            difficulty_stats[difficulty]['max_vote_correct'] += 1
        if item["upper_bound_correctness"] == 1:
            difficulty_stats[difficulty]['upper_bound_correct'] += 1
    
    # 计算准确率
    difficulty_accuracy = {}
    for difficulty, stats in difficulty_stats.items():
        if stats['total'] > 0:
            difficulty_accuracy[difficulty] = {
                'max_vote_accuracy': stats['max_vote_correct'] / stats['total'],
                'upper_bound_accuracy': stats['upper_bound_correct'] / stats['total'],
                'total_questions': stats['total']
            }
    
    return difficulty_accuracy

def plot_database_comparison(db_accuracy, vis_dir):
    """绘制数据库对比图"""
    # 准备数据
    db_ids = list(db_accuracy.keys())
    max_vote_acc = [db_accuracy[db_id]['max_vote_accuracy'] for db_id in db_ids]
    upper_bound_acc = [db_accuracy[db_id]['upper_bound_accuracy'] for db_id in db_ids]
    total_questions = [db_accuracy[db_id]['total_questions'] for db_id in db_ids]
    
    # 按总问题数排序
    sorted_indices = np.argsort(total_questions)[::-1]
    db_ids = [db_ids[i] for i in sorted_indices]
    max_vote_acc = [max_vote_acc[i] for i in sorted_indices]
    upper_bound_acc = [upper_bound_acc[i] for i in sorted_indices]
    total_questions = [total_questions[i] for i in sorted_indices]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 第一个子图：准确率对比
    x = np.arange(len(db_ids))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, max_vote_acc, width, label='Max Vote', alpha=0.7, color='skyblue')
    bars2 = ax1.bar(x + width/2, upper_bound_acc, width, label='Upper Bound', alpha=0.7, color='lightcoral')
    
    ax1.set_xlabel('Database')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Max Vote vs Upper Bound Accuracy by Database')
    ax1.set_xticks(x)
    ax1.set_xticklabels(db_ids, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, (mv, ub) in enumerate(zip(max_vote_acc, upper_bound_acc)):
        ax1.text(i - width/2, mv + 0.01, f'{mv:.2f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, ub + 0.01, f'{ub:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 第二个子图：问题数量
    bars3 = ax2.bar(x, total_questions, alpha=0.7, color='lightgreen')
    ax2.set_xlabel('Database')
    ax2.set_ylabel('Number of Questions')
    ax2.set_title('Number of Questions per Database')
    ax2.set_xticks(x)
    ax2.set_xticklabels(db_ids, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 在柱状图上添加数量标签
    for i, count in enumerate(total_questions):
        ax2.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{vis_dir}/database_comparison_r1.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_difficulty_comparison(difficulty_accuracy, vis_dir):
    """绘制难度对比图"""
    # 按难度顺序排序（如果有标准顺序）
    difficulty_order = ['simple', 'moderate', 'challenging', 'unknown']
    difficulties = [d for d in difficulty_order if d in difficulty_accuracy]
    
    max_vote_acc = [difficulty_accuracy[d]['max_vote_accuracy'] for d in difficulties]
    upper_bound_acc = [difficulty_accuracy[d]['upper_bound_accuracy'] for d in difficulties]
    total_questions = [difficulty_accuracy[d]['total_questions'] for d in difficulties]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 第一个子图：准确率对比
    x = np.arange(len(difficulties))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, max_vote_acc, width, label='Max Vote', alpha=0.7, color='skyblue')
    bars2 = ax1.bar(x + width/2, upper_bound_acc, width, label='Upper Bound', alpha=0.7, color='lightcoral')
    
    ax1.set_xlabel('Difficulty Level')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Difficulty Level')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in difficulties])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, (mv, ub) in enumerate(zip(max_vote_acc, upper_bound_acc)):
        ax1.text(i - width/2, mv + 0.01, f'{mv:.3f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, ub + 0.01, f'{ub:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 第二个子图：问题数量分布
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    ax2.pie(total_questions, labels=[d.capitalize() for d in difficulties], autopct='%1.1f%%', 
            colors=colors[:len(difficulties)])
    ax2.set_title('Question Distribution by Difficulty')
        
    plt.tight_layout()
    plt.savefig(f'{vis_dir}/difficulty_comparison_r1.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_statistics(diff_data, db_accuracy, difficulty_accuracy):
    """打印统计信息"""
    print("=" * 60)
    print("DATABASE ACCURACY STATISTICS")
    print("=" * 60)
    
    # 数据库统计
    df_db = pd.DataFrame([
        {
            'Database': db_id,
            'Max Vote Accuracy': stats['max_vote_accuracy'],
            'Upper Bound Accuracy': stats['upper_bound_accuracy'],
            'Improvement': stats['upper_bound_accuracy'] - stats['max_vote_accuracy'],
            'Total Questions': stats['total_questions']
        }
        for db_id, stats in db_accuracy.items()
    ])
    
    print(df_db.round(4))
    
    print("\n" + "=" * 60)
    print("DIFFICULTY LEVEL STATISTICS")
    print("=" * 60)
    
    # 难度统计
    df_diff = pd.DataFrame([
        {
            'Difficulty': difficulty.capitalize(),
            'Max Vote Accuracy': stats['max_vote_accuracy'],
            'Upper Bound Accuracy': stats['upper_bound_accuracy'],
            'Improvement': stats['upper_bound_accuracy'] - stats['max_vote_accuracy'],
            'Total Questions': stats['total_questions']
        }
        for difficulty, stats in difficulty_accuracy.items()
    ])
    
    print(df_diff.round(4))
    
    # 总体统计 - 修正：基于0/1标签计算
    total_max_vote = sum(1 for item in diff_data if item["max_vote_correctness"] == 1)
    total_upper_bound = sum(1 for item in diff_data if item["upper_bound_correctness"] == 1)
    total_questions = len(diff_data)
    
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Total Questions: {total_questions}")
    print(f"Max Vote Accuracy: {total_max_vote/total_questions:.4f} ({total_max_vote}/{total_questions})")
    print(f"Upper Bound Accuracy: {total_upper_bound/total_questions:.4f} ({total_upper_bound}/{total_questions})")
    print(f"Overall Improvement: {(total_upper_bound - total_max_vote)/total_questions:.4f}")


if __name__ == "__main__":
    # 加载数据
    diff_data, id_to_difficulty = load_data()
    
    # 计算准确率
    db_accuracy = calculate_database_accuracy(diff_data, id_to_difficulty)
    difficulty_accuracy = calculate_difficulty_accuracy(diff_data, id_to_difficulty)
    
    # 打印统计信息
    print_statistics(diff_data, db_accuracy, difficulty_accuracy)
    vis_dir = "vis_results/r1"
    os.makedirs(vis_dir, exist_ok=True)
    
    # 绘制图表
    plot_database_comparison(db_accuracy, vis_dir)
    plot_difficulty_comparison(difficulty_accuracy, vis_dir)