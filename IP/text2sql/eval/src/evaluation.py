import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
import csv
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
matplotlib.use('Agg')  # 使用非交互式后端

def load_json(dir):
    with open(dir, 'r') as j:
        contents = json.loads(j.read())
    return contents

def result_callback(result):
    exec_result.append(result)

def save_detailed_comparison(exec_result, sql_keys, diff_json_path, output_path):
    """保存详细的比较结果到CSV"""
    contents = load_json(diff_json_path)
    content_dict = {str(item['question_id']): item for item in contents}
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([
            'question_id', 'db_name', 'difficulty', 'is_correct',
            'gt_length', 'pred_length', 'element_overlap_rate',
            'gt_answer_preview', 'pred_answer_preview', 'analysis'
        ])
        
        for i, result in enumerate(exec_result):
            if i < len(sql_keys):
                question_id = sql_keys[i]
                if question_id in content_dict:
                    content = content_dict[question_id]
                    db_name = content['db_id']
                    difficulty = content['difficulty']
                    
                    # 获取结果
                    gt_answer = result['ground_truth_answer']
                    pred_answer = result['predicted_answer']
                    
                    # 转换为字符串用于预览
                    def truncate_preview(obj, max_length=100):
                        preview = str(obj)
                        return preview[:max_length] + '...' if len(preview) > max_length else preview
                    
                    gt_preview = truncate_preview(gt_answer)
                    pred_preview = truncate_preview(pred_answer)
                    
                    # 计算元素重叠率
                    def calculate_overlap_rate(gt, pred):
                        if not isinstance(gt, list) or not isinstance(pred, list):
                            return 0
                        
                        # 转换为字符串集合进行比较
                        gt_set = set(str(x) for x in gt)
                        pred_set = set(str(x) for x in pred)
                        
                        intersection = gt_set & pred_set
                        union = gt_set | pred_set
                        
                        return len(intersection) / len(union) if union else 0
                    
                    # 计算位置匹配率
                    def calculate_position_match_rate(gt, pred):
                        if not isinstance(gt, list) or not isinstance(pred, list):
                            return 0
                        
                        min_len = min(len(gt), len(pred))
                        if min_len == 0:
                            return 0
                        
                        match_count = 0
                        for i in range(min_len):
                            if str(gt[i]) == str(pred[i]):
                                match_count += 1
                        
                        return match_count / min_len
                    
                    overlap_rate = calculate_overlap_rate(gt_answer, pred_answer)
                    position_match_rate = calculate_position_match_rate(gt_answer, pred_answer)
                    
                    # 生成分析信息
                    analysis_parts = []
                    analysis_parts.append(f"GT长度:{len(gt_answer) if isinstance(gt_answer, list) else 1}")
                    analysis_parts.append(f"Pred长度:{len(pred_answer) if isinstance(pred_answer, list) else 1}")
                    analysis_parts.append(f"重叠率:{overlap_rate:.2f}")
                    analysis_parts.append(f"位置匹配率:{position_match_rate:.2f}")
                    
                    writer.writerow([
                        question_id,
                        db_name,
                        difficulty,
                        '正确' if result['res'] == 1 else '错误',
                        len(gt_answer) if isinstance(gt_answer, list) else 1,
                        len(pred_answer) if isinstance(pred_answer, list) else 1,
                        f"{overlap_rate:.2f}",
                        gt_preview,
                        pred_preview,
                        '; '.join(analysis_parts)
                    ])
    
    print(f"Detailed comparison saved to: {output_path}")

def save_detailed_stats(exec_result, sql_keys, diff_json_path, output_path):
    """保存详细的统计信息到TXT文件"""
    contents = load_json(diff_json_path)
    content_dict = {str(item['question_id']): item for item in contents}
    
    db_stats = {}
    
    for i, result in enumerate(exec_result):
        if i < len(sql_keys):
            question_id = sql_keys[i]
            if question_id in content_dict:
                db_name = content_dict[question_id]['db_id']
                difficulty = content_dict[question_id]['difficulty']
                
                if db_name not in db_stats:
                    db_stats[db_name] = {
                        'simple': {'correct': [], 'wrong': [], 'total': 0},
                        'moderate': {'correct': [], 'wrong': [], 'total': 0},
                        'challenging': {'correct': [], 'wrong': [], 'total': 0}
                    }
                
                if result['res'] == 1:
                    db_stats[db_name][difficulty]['correct'].append(question_id)
                else:
                    db_stats[db_name][difficulty]['wrong'].append(question_id)
                db_stats[db_name][difficulty]['total'] += 1
    
    # 写入TXT文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Database Performance Statistics (Compact Format)\n")
        f.write("=" * 80 + "\n\n")
        
        for db_name, difficulties in db_stats.items():
            for diff in ['simple', 'moderate', 'challenging']:
                data = difficulties[diff]
                if data['total'] > 0:  # 只输出有数据的难度级别
                    total = data['total']
                    correct_count = len(data['correct'])
                    accuracy = (correct_count / total * 100) if total > 0 else 0
                    
                    f.write(f"{db_name} - {diff}: Total={total}, Correct={correct_count}, Accuracy={accuracy:.2f}%\n")
                    f.write(f"Correct IDs: {data['correct']}\n")
                    f.write(f"Wrong IDs: {data['wrong']}\n")
                    f.write("\n")
    
    print(f"Detailed statistics saved to: {output_path}")
    
    return db_stats

def plot_accuracy_analysis(exec_result, sql_keys, diff_json_path, output_dir):
    """绘制准确度分析图：热力图和分布图"""
    contents = load_json(diff_json_path)
    content_dict = {str(item['question_id']): item for item in contents}
    
    # 收集数据
    db_difficulty_data = defaultdict(lambda: defaultdict(list))
    db_performance = defaultdict(lambda: {'total': 0, 'correct': 0})
    difficulty_performance = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for i, result in enumerate(exec_result):
        if i < len(sql_keys):
            question_id = sql_keys[i]
            if question_id in content_dict:
                content = content_dict[question_id]
                db_name = content['db_id']
                difficulty = content['difficulty']
                
                # 收集数据库和难度级别的性能数据
                db_difficulty_data[db_name][difficulty].append(result['res'])
                db_performance[db_name]['total'] += 1
                db_performance[db_name]['correct'] += result['res']
                difficulty_performance[difficulty]['total'] += 1
                difficulty_performance[difficulty]['correct'] += result['res']
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 子图1：热力图 - 数据库 vs 难度级别的准确率
    if db_difficulty_data:
        # 准备热力图数据
        db_names = sorted(db_difficulty_data.keys())
        difficulties = ['simple', 'moderate', 'challenging']
        heatmap_data = []
        
        for db_name in db_names:
            row = []
            for diff in difficulties:
                results = db_difficulty_data[db_name].get(diff, [])
                if results:
                    accuracy = sum(results) / len(results) * 100
                else:
                    accuracy = 0
                row.append(accuracy)
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        # 绘制热力图
        im = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # 设置坐标轴
        ax1.set_xticks(range(len(difficulties)))
        ax1.set_xticklabels([d.capitalize() for d in difficulties])
        ax1.set_yticks(range(len(db_names)))
        ax1.set_yticklabels(db_names)
        
        # 添加数值标注
        for i in range(len(db_names)):
            for j in range(len(difficulties)):
                text = ax1.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                               ha="center", va="center", color="black", fontsize=10,
                               fontweight='bold' if heatmap_data[i, j] > 50 else 'normal')
        
        ax1.set_title('Accuracy Heatmap: Database vs Difficulty', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Difficulty Level', fontsize=12)
        ax1.set_ylabel('Database', fontsize=12)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Accuracy (%)', fontsize=12)
    
    # 子图2：分布图 - 准确率分布
    if db_performance:
        # 计算每个数据库的准确率
        db_accuracies = []
        db_labels = []
        
        for db_name, perf in db_performance.items():
            if perf['total'] > 0:
                accuracy = (perf['correct'] / perf['total']) * 100
                db_accuracies.append(accuracy)
                db_labels.append(f"{db_name}\n({perf['correct']}/{perf['total']})")
        
        # 绘制柱状图
        bars = ax2.bar(range(len(db_accuracies)), db_accuracies, color='skyblue', edgecolor='black', alpha=0.7)
        
        # 设置坐标轴
        ax2.set_xticks(range(len(db_accuracies)))
        ax2.set_xticklabels(db_labels, rotation=45, ha='right')
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_ylim(0, 100)
        
        # 在柱子上添加数值
        for bar, acc in zip(bars, db_accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 添加难度级别的平均准确率
        difficulty_accuracies = []
        difficulty_labels = []
        for diff in ['simple', 'moderate', 'challenging']:
            if difficulty_performance[diff]['total'] > 0:
                accuracy = (difficulty_performance[diff]['correct'] / difficulty_performance[diff]['total']) * 100
                difficulty_accuracies.append(accuracy)
                difficulty_labels.append(diff.capitalize())
        
        # 在右侧添加难度级别的准确率
        ax2_right = ax2.twinx()
        ax2_right.plot(range(len(difficulty_accuracies)), difficulty_accuracies, 
                      'ro-', linewidth=3, markersize=8, label='Difficulty Avg')
        ax2_right.set_ylabel('Difficulty Average Accuracy (%)', fontsize=12, color='red')
        ax2_right.set_ylim(0, 100)
        ax2_right.tick_params(axis='y', labelcolor='red')
        ax2_right.legend(loc='upper right')
        
        ax2.set_title('Accuracy Distribution by Database', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = f"{output_dir}/accuracy_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Accuracy analysis plot saved to: {output_path}")
    
    # 保存统计数据
    stats = {
        'database_performance': {
            db_name: {
                'total': perf['total'],
                'correct': perf['correct'],
                'accuracy': (perf['correct'] / perf['total'] * 100) if perf['total'] > 0 else 0
            } for db_name, perf in db_performance.items()
        },
        'difficulty_performance': {
            diff: {
                'total': perf['total'],
                'correct': perf['correct'],
                'accuracy': (perf['correct'] / perf['total'] * 100) if perf['total'] > 0 else 0
            } for diff, perf in difficulty_performance.items()
        }
    }
    
    stats_path = f"{output_dir}/accuracy_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Accuracy statistics saved to: {stats_path}")

def execute_sql(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    
    # 处理结果格式
    def process_result(result):
        if not result:
            return []
        if len(result[0]) == 1:
            return [row[0] for row in result]
        else:
            return [list(row) for row in result]
    
    predicted_processed = process_result(predicted_res)
    ground_truth_processed = process_result(ground_truth_res)
    
    return {
        'res': res,
        'predicted_answer': predicted_processed,
        'ground_truth_answer': ground_truth_processed,
        'predicted_sql': predicted_sql,
        'ground_truth_sql': ground_truth,
        # 原始结果用于详细比较
        'predicted_raw': predicted_res,
        'ground_truth_raw': ground_truth_res
    }

def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    try:
        detailed_result = func_timeout(meta_time_out, execute_sql,
                                  args=(predicted_sql, ground_truth, db_place))
        result = {
            'sql_idx': idx, 
            'res': detailed_result['res'],
            'predicted_answer': detailed_result['predicted_answer'],
            'ground_truth_answer': detailed_result['ground_truth_answer'],
            'predicted_sql': detailed_result['predicted_sql'],
            'ground_truth_sql': detailed_result['ground_truth_sql']
        }
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = {
            'sql_idx': idx, 
            'res': 0,
            'predicted_answer': 'timeout',
            'ground_truth_answer': 'timeout',
            'predicted_sql': predicted_sql,
            'ground_truth_sql': ground_truth
        }
    except Exception as e:
        result = {
            'sql_idx': idx, 
            'res': 0,
            'predicted_answer': f'error: {str(e)}',
            'ground_truth_answer': f'error: {str(e)}',
            'predicted_sql': predicted_sql,
            'ground_truth_sql': ground_truth
        }
    return result

def package_sqls(pred_sql_path, gt_sql_path, db_root_path, diff_json_path=None, mode='gpt', data_mode='dev'):
    clean_sqls = []
    db_path_list = []
    sql_data = json.load(open(pred_sql_path + 'predict_' + data_mode + '.json', 'r'))
    
    # 加载diff_json并创建question_id集合
    valid_question_ids = set()
    if diff_json_path:
        contents = load_json(diff_json_path)
        valid_question_ids = {str(item['question_id']) for item in contents}
    
    # 过滤sql_keys：只保留在diff_json中存在的question_id
    sql_keys = []
    for key in sql_data.keys():
        # 如果有diff_json_path，只保留存在的question_id；如果没有，保留所有
        if not diff_json_path or key in valid_question_ids:
            sql_keys.append(key)
    
    # 按数字排序
    sql_keys = sorted(sql_keys, key=lambda x: int(x))
    print(f"Filtered SQL keys count: {len(sql_keys)}")
    
    if mode == 'gpt':
        for idx in sql_keys:
            sql_str = sql_data[idx]
            if type(sql_str) == str:
                sql, db_name = sql_str.split('\t----- bird -----\t')
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    elif mode == 'gt':
        sqls = open(gt_sql_path + data_mode + '.sql')
        sql_txt = sqls.readlines()
        # 只选择过滤后的sql_keys对应的行
        sql_txt = [sql_txt[int(i)] for i in sql_keys]
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list, sql_keys

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i,sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out), callback=result_callback)
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_diff(exec_results, diff_json_path, pred_sql_path, data_mode, sql_keys):
    num_queries = len(exec_results)
    contents = load_json(diff_json_path)
    
    # 创建内容字典，按question_id索引
    content_dict = {str(item['question_id']): item for item in contents}
    
    # 按过滤后的sql_keys顺序重新组织contents
    filtered_contents = []
    for key in sql_keys:
        if key in content_dict:
            filtered_contents.append(content_dict[key])
    
    simple_results, moderate_results, challenging_results = [], [], []

    for i, content in enumerate(filtered_contents):
        if content['difficulty'] == 'simple':
            simple_results.append(exec_results[i])
        elif content['difficulty'] == 'moderate':
            moderate_results.append(exec_results[i])
        elif content['difficulty'] == 'challenging':
            challenging_results.append(exec_results[i])
            
    total_correct = sum([res['res'] for res in simple_results+moderate_results+challenging_results])
    total_sqls = len(simple_results+moderate_results+challenging_results)
    
    simple_acc = sum([res['res'] for res in simple_results])/len(simple_results) if simple_results else 0
    moderate_acc = sum([res['res'] for res in moderate_results])/len(moderate_results) if moderate_results else 0
    challenging_acc = sum([res['res'] for res in challenging_results])/len(challenging_results) if challenging_results else 0
    all_acc = total_correct/total_sqls if total_sqls > 0 else 0
    
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), total_sqls]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists

def save_exec_result(exec_result, sql_keys, output_path):
    """保存详细的执行结果到JSON文件"""
    result_dict = {}
    for i, result in enumerate(exec_result):
        if i < len(sql_keys):
            question_id = sql_keys[i]
            result_dict[question_id] = {
                'sql_idx': result['sql_idx'],
                'res': result['res'],
                'predicted_answer': result['predicted_answer'],
                'ground_truth_answer': result['ground_truth_answer'],
                'predicted_sql': result['predicted_sql'],
                'ground_truth_sql': result['ground_truth_sql']
            }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    print(f"Detailed execution results saved to: {output_path}")

def print_data(score_lists,count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='')
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='')
    args_parser.add_argument('--data_mode', type=str, required=True, default='dev')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument('--num_cpus', type=int, default=1)
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    args_parser.add_argument('--mode_gt', type=str, default='gt')
    args_parser.add_argument('--mode_predict', type=str, default='gpt')
    args_parser.add_argument('--difficulty',type=str,default='simple')
    args_parser.add_argument('--diff_json_path',type=str,default='')
    args_parser.add_argument('--output_result', type=str, default='execution_results.json', help='Output file for execution results')
    args = args_parser.parse_args()
    exec_result = []

    # 在package_sqls阶段就进行过滤
    pred_queries, db_paths, filtered_sql_keys = package_sqls(
        args.predicted_sql_path, args.ground_truth_path, args.db_root_path, 
        diff_json_path=args.diff_json_path,  # 传入diff_json_path进行过滤
        mode=args.mode_predict, 
        data_mode=args.data_mode
    )
    
    # generate gt sqls (使用相同的过滤逻辑)
    gt_queries, db_paths_gt, _ = package_sqls(
        args.predicted_sql_path, args.ground_truth_path, args.db_root_path,
        diff_json_path=args.diff_json_path,  # 传入diff_json_path进行过滤
        mode='gt', 
        data_mode=args.data_mode
    )

    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)
    
    # 保存执行结果到JSON文件
    output_dir = args.predicted_sql_path
    os.makedirs(output_dir, exist_ok=True)
    output_result_path = os.path.join(output_dir, args.output_result)
    save_exec_result(exec_result, filtered_sql_keys, output_result_path)
        
    # 新增：保存详细统计
    stats_output_path = os.path.join(output_dir, 'detailed_statistics.txt')
    save_detailed_stats(exec_result, filtered_sql_keys, args.diff_json_path, stats_output_path)
    
    # 新增：保存详细比较CSV
    comparison_output_path = os.path.join(output_dir, 'detailed_comparison.csv')
    save_detailed_comparison(exec_result, filtered_sql_keys, args.diff_json_path, comparison_output_path)
    
    # 新增：绘制准确度分析图
    os.makedirs(output_dir+"vis",exist_ok=True)
    print("path: ",output_dir+"vis")
    plot_accuracy_analysis(exec_result, filtered_sql_keys, args.diff_json_path, output_dir+"vis")
    
    print('start calculate')
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = \
        compute_acc_by_diff(exec_result, args.diff_json_path, args.predicted_sql_path, args.data_mode, filtered_sql_keys)
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print("Finished evaluation")
    