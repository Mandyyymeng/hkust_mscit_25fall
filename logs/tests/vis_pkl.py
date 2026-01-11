import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Tuple

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_data(detailed_json_path: str, answers_json_path: str, dev_json_path: str) -> Tuple[Dict, Dict, Dict]:
    """加载数据"""
    with open(detailed_json_path, 'r', encoding='utf-8') as f:
        detailed_data = json.load(f)
    
    with open(answers_json_path, 'r', encoding='utf-8') as f:
        answers_data = json.load(f)
    
    # 新增：加载dev.json获取难度信息
    with open(dev_json_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    # 创建question_id到difficulty的映射
    id_to_difficulty = {}
    for item in dev_data:
        id_to_difficulty[item["question_id"]] = item["difficulty"]
    
    return detailed_data, answers_data, id_to_difficulty

def normalize_sql(sql: str) -> str:
    """标准化SQL字符串用于比较"""
    if not sql:
        return ""
    return ' '.join(sql.lower().split())

def compare_results(predicted_result, ground_truth_answer) -> Tuple[bool, bool]:
    """
    比较预测结果和真实答案
    返回: (完全匹配, 部分匹配)
    """
    if not predicted_result or not ground_truth_answer:
        return False, False
    
    # 提取预测结果
    pred_str = str(predicted_result).lower().strip()
    truth_str = str(ground_truth_answer).lower().strip()
    
    # 完全匹配
    exact_match = pred_str == truth_str
    
    # 部分匹配：预测结果包含真实答案
    partial_match = truth_str in pred_str
    
    return exact_match, partial_match

def analyze_data(detailed_data: Dict, answers_data: Dict, id_to_difficulty: Dict) -> Dict[str, Any]:
    """分析数据并计算指标"""
    
    analysis_results = {
        'db_stats': {},
        'difficulty_stats': {},  # 新增：难度统计
        'question_stats': {},
        'all_paths': []
    }
    
    # 初始化难度统计
    difficulty_stats = analysis_results['difficulty_stats']
    
    # 第一遍：收集所有路径信息
    for qid, question_data in detailed_data.items():
        db_id = question_data['db_id']
        question_id = question_data['question_id']
        ground_truth = answers_data.get(str(question_id))
        
        # 初始化数据库统计
        if db_id not in analysis_results['db_stats']:
            analysis_results['db_stats'][db_id] = {
                'total_questions': 0,
                'exact_match_upper': 0,
                'partial_match_upper': 0,
                'selected_exact_match': 0,
                'selected_partial_match': 0,
                'path_lengths': [],
                'consistency_scores': []
            }
        
        db_stats = analysis_results['db_stats'][db_id]
        db_stats['total_questions'] += 1
        
        # 初始化难度统计
        difficulty = id_to_difficulty.get(question_id, 'unknown')
        if difficulty not in difficulty_stats:
            difficulty_stats[difficulty] = {
                'total_questions': 0,
                'exact_match_upper': 0,
                'partial_match_upper': 0,
                'selected_exact_match': 0,
                'selected_partial_match': 0
            }
        
        diff_stats = difficulty_stats[difficulty]
        diff_stats['total_questions'] += 1
        
        # 处理每个路径
        selected_path_index = question_data.get('selected_path_index', 0)
        all_paths_info = question_data.get('all_paths_info', [])
        
        question_exact_match_upper = False
        question_partial_match_upper = False
        selected_exact_match = False
        selected_partial_match = False
        
        for path_info in all_paths_info:
            path_index = path_info['path_index']
            execution_result = path_info.get('execution_result', {})
            result_data = execution_result.get('result', [])
            
            # 提取第一个结果
            first_result = result_data[0][0] if result_data and len(result_data) > 0 and len(result_data[0]) > 0 else None
            
            # 与真实答案比较
            exact_match, partial_match = compare_results(first_result, ground_truth)
            
            # 更新upper bound
            if exact_match:
                question_exact_match_upper = True
            if partial_match:
                question_partial_match_upper = True
            
            # 更新selected path
            if path_index == selected_path_index:
                selected_exact_match = exact_match
                selected_partial_match = partial_match
            
            # 收集路径数据
            path_data = {
                'db_id': db_id,
                'question_id': question_id,
                'path_index': path_index,
                'path_length': path_info.get('path_length', 0),
                'consistency_score': path_info.get('consistency_score', 0),
                'is_selected': path_index == selected_path_index,
                'exact_match': exact_match,
                'partial_match': partial_match,
                'difficulty': difficulty  # 新增难度字段
            }
            analysis_results['all_paths'].append(path_data)
        
        # 更新数据库统计
        if question_exact_match_upper:
            db_stats['exact_match_upper'] += 1
        if question_partial_match_upper:
            db_stats['partial_match_upper'] += 1
        if selected_exact_match:
            db_stats['selected_exact_match'] += 1
        if selected_partial_match:
            db_stats['selected_partial_match'] += 1
        
        db_stats['path_lengths'].extend([p['path_length'] for p in all_paths_info])
        db_stats['consistency_scores'].extend([p['consistency_score'] for p in all_paths_info])
        
        # 更新难度统计（与db_stats相同的逻辑）
        if question_exact_match_upper:
            diff_stats['exact_match_upper'] += 1
        if question_partial_match_upper:
            diff_stats['partial_match_upper'] += 1
        if selected_exact_match:
            diff_stats['selected_exact_match'] += 1
        if selected_partial_match:
            diff_stats['selected_partial_match'] += 1
    
    return analysis_results
          
def create_visualizations(analysis_results: Dict, output_dir: str):
    """创建可视化图表"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    all_paths_df = pd.DataFrame(analysis_results['all_paths'])
    db_stats = analysis_results['db_stats']
    difficulty_stats = analysis_results['difficulty_stats']  # 新增：难度统计
    
    # 1. 数据库准确率比较柱状图
    db_names = list(db_stats.keys())
    exact_match_upper = [db_stats[db]['exact_match_upper'] / db_stats[db]['total_questions'] for db in db_names]
    partial_match_upper = [db_stats[db]['partial_match_upper'] / db_stats[db]['total_questions'] for db in db_names]
    selected_exact_match = [db_stats[db]['selected_exact_match'] / db_stats[db]['total_questions'] for db in db_names]
    selected_partial_match = [db_stats[db]['selected_partial_match'] / db_stats[db]['total_questions'] for db in db_names]
    
    x = np.arange(len(db_names))
    width = 0.2
    
    plt.figure(figsize=(16, 10))
    bars1 = plt.bar(x - 1.5*width, exact_match_upper, width, label='Exact Match Upper Bound', alpha=0.8, color='skyblue')
    bars2 = plt.bar(x - 0.5*width, partial_match_upper, width, label='Partial Match Upper Bound', alpha=0.8, color='lightcoral')
    bars3 = plt.bar(x + 0.5*width, selected_exact_match, width, label='Selected Exact Match', alpha=0.8, color='lightgreen')
    bars4 = plt.bar(x + 1.5*width, selected_partial_match, width, label='Selected Partial Match', alpha=0.8, color='gold')
    
    # 在柱子顶端添加准确率文本
    for bars, values in zip([bars1, bars2, bars3, bars4], 
                           [exact_match_upper, partial_match_upper, selected_exact_match, selected_partial_match]):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.xlabel('Database', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Comparison by Database', fontsize=16, fontweight='bold')
    plt.xticks(x, db_names, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison_by_db.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 新增：难度准确率比较柱状图
    difficulties = list(difficulty_stats.keys())
    
    # 按难度排序（如果有标准顺序）
    difficulty_order = ['simple', 'moderate', 'challenging', 'unknown']
    difficulties = [d for d in difficulty_order if d in difficulties] + [d for d in difficulties if d not in difficulty_order]
    
    exact_match_upper_diff = [difficulty_stats[diff]['exact_match_upper'] / difficulty_stats[diff]['total_questions'] for diff in difficulties]
    partial_match_upper_diff = [difficulty_stats[diff]['partial_match_upper'] / difficulty_stats[diff]['total_questions'] for diff in difficulties]
    selected_exact_match_diff = [difficulty_stats[diff]['selected_exact_match'] / difficulty_stats[diff]['total_questions'] for diff in difficulties]
    selected_partial_match_diff = [difficulty_stats[diff]['selected_partial_match'] / difficulty_stats[diff]['total_questions'] for diff in difficulties]
    
    x_diff = np.arange(len(difficulties))
    width_diff = 0.2
    
    plt.figure(figsize=(14, 8))
    bars1_diff = plt.bar(x_diff - 1.5*width_diff, exact_match_upper_diff, width_diff, label='Exact Match Upper Bound', alpha=0.8, color='skyblue')
    bars2_diff = plt.bar(x_diff - 0.5*width_diff, partial_match_upper_diff, width_diff, label='Partial Match Upper Bound', alpha=0.8, color='lightcoral')
    bars3_diff = plt.bar(x_diff + 0.5*width_diff, selected_exact_match_diff, width_diff, label='Selected Exact Match', alpha=0.8, color='lightgreen')
    bars4_diff = plt.bar(x_diff + 1.5*width_diff, selected_partial_match_diff, width_diff, label='Selected Partial Match', alpha=0.8, color='gold')
    
    # 在柱子顶端添加准确率文本
    for bars, values in zip([bars1_diff, bars2_diff, bars3_diff, bars4_diff], 
                           [exact_match_upper_diff, partial_match_upper_diff, selected_exact_match_diff, selected_partial_match_diff]):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.xlabel('Difficulty Level', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Comparison by Difficulty Level', fontsize=16, fontweight='bold')
    plt.xticks(x_diff, [d.capitalize() for d in difficulties])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, max(max(exact_match_upper_diff), max(partial_match_upper_diff), 
                   max(selected_exact_match_diff), max(selected_partial_match_diff)) + 0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison_by_difficulty.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 路径长度与一致性得分的关系图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=all_paths_df, x='path_length', y='consistency_score', 
                   alpha=0.6, s=60)
    plt.title('Path Length vs Consistency Score', fontsize=16, fontweight='bold')
    plt.xlabel('Path Length')
    plt.ylabel('Consistency Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'path_length_vs_consistency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 按数据库分组的路径长度与一致性得分
    plt.figure(figsize=(15, 10))
    sns.scatterplot(data=all_paths_df, x='path_length', y='consistency_score', 
                   hue='db_id', alpha=0.7, s=60)
    plt.title('Path Length vs Consistency Score by Database', fontsize=16, fontweight='bold')
    plt.xlabel('Path Length')
    plt.ylabel('Consistency Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'path_length_vs_consistency_by_db.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 总体统计热力图
    overall_stats = []
    for db in db_names:
        stats = db_stats[db]
        overall_stats.append({
            'Database': db,
            'Total Questions': stats['total_questions'],
            'Exact Match Upper': stats['exact_match_upper'] / stats['total_questions'],
            'Partial Match Upper': stats['partial_match_upper'] / stats['total_questions'],
            'Selected Exact Match': stats['selected_exact_match'] / stats['total_questions'],
            'Selected Partial Match': stats['selected_partial_match'] / stats['total_questions'],
            'Avg Path Length': np.mean(stats['path_lengths']),
            'Avg Consistency Score': np.mean(stats['consistency_scores'])
        })
    
    overall_df = pd.DataFrame(overall_stats)
    
    # 热力图数据准备
    heatmap_data = overall_df.set_index('Database')[['Exact Match Upper', 'Partial Match Upper', 
                                                    'Selected Exact Match', 'Selected Partial Match']]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Accuracy'})
    plt.title('Accuracy Metrics Heatmap by Database', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 路径长度分布
    plt.figure(figsize=(12, 8))
    sns.histplot(data=all_paths_df, x='path_length', bins=30, kde=True)
    plt.title('Distribution of Path Lengths', fontsize=16, fontweight='bold')
    plt.xlabel('Path Length')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'path_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 一致性得分分布
    plt.figure(figsize=(12, 8))
    sns.histplot(data=all_paths_df, x='consistency_score', bins=30, kde=True)
    plt.title('Distribution of Consistency Scores', fontsize=16, fontweight='bold')
    plt.xlabel('Consistency Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'consistency_score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return overall_df

def save_csv_results(analysis_results: Dict, overall_df: pd.DataFrame, output_dir: str):
    """保存CSV结果"""
    
    # 保存总体统计
    overall_df.to_csv(os.path.join(output_dir, 'overall_statistics.csv'), index=False)
    
    # 保存所有路径数据
    all_paths_df = pd.DataFrame(analysis_results['all_paths'])
    all_paths_df.to_csv(os.path.join(output_dir, 'all_paths_data.csv'), index=False)
    
    # 保存数据库级别统计
    db_stats_list = []
    for db, stats in analysis_results['db_stats'].items():
        db_stats_list.append({
            'Database': db,
            'Total Questions': stats['total_questions'],
            'Exact Match Upper Bound': stats['exact_match_upper'],
            'Partial Match Upper Bound': stats['partial_match_upper'],
            'Selected Exact Match': stats['selected_exact_match'],
            'Selected Partial Match': stats['selected_partial_match'],
            'Exact Match Upper Accuracy': stats['exact_match_upper'] / stats['total_questions'],
            'Partial Match Upper Accuracy': stats['partial_match_upper'] / stats['total_questions'],
            'Selected Exact Match Accuracy': stats['selected_exact_match'] / stats['total_questions'],
            'Selected Partial Match Accuracy': stats['selected_partial_match'] / stats['total_questions'],
            'Average Path Length': np.mean(stats['path_lengths']),
            'Average Consistency Score': np.mean(stats['consistency_scores'])
        })
    
    db_stats_df = pd.DataFrame(db_stats_list)
    db_stats_df.to_csv(os.path.join(output_dir, 'database_statistics.csv'), index=False)

def main():
    # 文件路径配置
    detailed_json_path = 'logs/pred_sqls/pred_sqls_qwen32b_bird_300_detailed.json'
    answers_json_path = 'data/bird/dev/dev_answer.json'
    dev_json_path = 'data/bird/dev/dev_all.json'  # 新增：dev.json路径
    output_dir = 'vis_results'
    
    print("🚀 开始数据分析...")
    
    # 加载数据（修改为3个参数）
    print("📂 加载数据...")
    detailed_data, answers_data, id_to_difficulty = load_data(detailed_json_path, answers_json_path, dev_json_path)
    
    # 分析数据（修改为3个参数）
    print("📊 分析数据...")
    analysis_results = analyze_data(detailed_data, answers_data, id_to_difficulty)
    
    # 创建可视化
    print("🎨 创建可视化图表...")
    overall_df = create_visualizations(analysis_results, output_dir)
    
    # 保存CSV结果
    print("💾 保存结果...")
    save_csv_results(analysis_results, overall_df, output_dir)
    
    # 打印总体统计
    print("\n📈 总体统计:")
    print(f"总数据库数量: {len(analysis_results['db_stats'])}")
    print(f"总问题数量: {len(detailed_data)}")
    print(f"总路径数量: {len(analysis_results['all_paths'])}")
    
    # 计算总体准确率
    total_questions = sum([stats['total_questions'] for stats in analysis_results['db_stats'].values()])
    total_exact_upper = sum([stats['exact_match_upper'] for stats in analysis_results['db_stats'].values()])
    total_partial_upper = sum([stats['partial_match_upper'] for stats in analysis_results['db_stats'].values()])
    total_selected_exact = sum([stats['selected_exact_match'] for stats in analysis_results['db_stats'].values()])
    total_selected_partial = sum([stats['selected_partial_match'] for stats in analysis_results['db_stats'].values()])
    
    print(f"\n🎯 总体准确率:")
    print(f"Exact Match Upper Bound: {total_exact_upper}/{total_questions} ({total_exact_upper/total_questions:.3f})")
    print(f"Partial Match Upper Bound: {total_partial_upper}/{total_questions} ({total_partial_upper/total_questions:.3f})")
    print(f"Selected Exact Match: {total_selected_exact}/{total_questions} ({total_selected_exact/total_questions:.3f})")
    print(f"Selected Partial Match: {total_selected_partial}/{total_questions} ({total_selected_partial/total_questions:.3f})")
    
    # 新增：难度统计
    difficulty_stats = analysis_results['difficulty_stats']
    print(f"\n📊 难度统计:")
    for difficulty, stats in difficulty_stats.items():
        total = stats['total_questions']
        if total > 0:
            exact_acc = stats['exact_match_upper'] / total
            partial_acc = stats['partial_match_upper'] / total
            selected_exact_acc = stats['selected_exact_match'] / total
            selected_partial_acc = stats['selected_partial_match'] / total
            print(f"{difficulty.capitalize()}: {total} questions")
            print(f"  Exact Upper: {exact_acc:.3f}, Partial Upper: {partial_acc:.3f}")
            print(f"  Selected Exact: {selected_exact_acc:.3f}, Selected Partial: {selected_partial_acc:.3f}")
    
    print(f"\n✅ 分析完成！结果保存在: {output_dir}/")

if __name__ == "__main__":
    main()