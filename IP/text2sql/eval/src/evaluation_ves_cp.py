import os
import sys
import json
import numpy as np
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
import time
import math
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import defaultdict
matplotlib.use('Agg') 

def load_json(dir):
    with open(dir, 'r') as j:
        contents = json.loads(j.read())
    return contents

def result_callback(result):
    exec_result.append(result)

def clean_abnormal(input):
    input = np.asarray(input)
    processed_list = []
    mean = np.mean(input,axis=0)
    std = np.std(input,axis=0)
    for x in input:
        if x < mean + 3 * std and x > mean - 3 * std:
            processed_list.append(x)
    return processed_list

def execute_sql(sql, db_path):
    # Connect to the database
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    start_time = time.time()
    cursor.execute(sql)
    exec_time = time.time() - start_time
    return exec_time

def iterated_execute_sql(predicted_sql,ground_truth,db_path,iterate_num):
    conn = sqlite3.connect(db_path)
    diff_list = []
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    time_ratio = 0
    if set(predicted_res) == set(ground_truth_res):
        # 保存每次执行的时间用于分析
        predicted_times = []
        ground_truth_times = []
        for i in range(iterate_num):
            predicted_time = execute_sql(predicted_sql, db_path)
            ground_truth_time = execute_sql(ground_truth, db_path)
            predicted_times.append(predicted_time)
            ground_truth_times.append(ground_truth_time)
            diff_list.append(ground_truth_time / predicted_time)
        processed_diff_list = clean_abnormal(diff_list)
        time_ratio = sum(processed_diff_list) / len(processed_diff_list)
        return time_ratio, predicted_times, ground_truth_times
    return time_ratio, [], []

def execute_model(predicted_sql,ground_truth, db_place, idx, iterate_num, meta_time_out):
    try:
        time_ratio, pred_times, gt_times = func_timeout(meta_time_out * iterate_num, iterated_execute_sql,
                                  args=(predicted_sql, ground_truth, db_place, iterate_num))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        time_ratio, pred_times, gt_times = 0, [], []
    except Exception as e:
        time_ratio, pred_times, gt_times = 0, [], []
    
    # 更健壮的数据库名称提取
    db_path_parts = db_place.split('/')
    db_name = db_path_parts[-2] if len(db_path_parts) >= 2 else 'unknown'
    
    result = {
        'sql_idx': idx, 
        'time_ratio': time_ratio,
        'predicted_times': pred_times,
        'ground_truth_times': gt_times,
        'db_name': db_name
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

def run_sqls_parallel(sqls, db_places, num_cpus=1, iterate_num=100, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i,sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, iterate_num, meta_time_out), callback=result_callback)
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_ves(exec_results):
    num_queries = len(exec_results)
    total_ratio = 0
    count = 0

    for i, result in enumerate(exec_results):
        if result['time_ratio'] != 0:
            count += 1
        total_ratio += math.sqrt(result['time_ratio']) * 100
    ves = (total_ratio/num_queries)
    return ves

def compute_ves_by_diff(exec_results, diff_json_path, pred_sql_path, data_mode, sql_keys):
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
    
    # 计算各难度级别的VES
    simple_ves = compute_ves(simple_results) if simple_results else 0
    moderate_ves = compute_ves(moderate_results) if moderate_results else 0
    challenging_ves = compute_ves(challenging_results) if challenging_results else 0
    all_ves = compute_ves(exec_results)
    
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_ves, moderate_ves, challenging_ves, all_ves, count_lists

def plot_advanced_time_analysis(exec_result, diff_json_path, sql_keys, output_dir):
    """绘制高级时间分析图：箱型图和热力图"""
    contents = load_json(diff_json_path)
    content_dict = {str(item['question_id']): item for item in contents}
    
    # 按数据库和难度分组数据
    db_difficulty_data = defaultdict(lambda: defaultdict(list))
    db_query_times = defaultdict(list)
    
    for i, result in enumerate(exec_result):
        if i < len(sql_keys):
            question_id = sql_keys[i]
            if question_id in content_dict:
                db_name = result['db_name']
                difficulty = content_dict[question_id]['difficulty']
                
                # 收集预测SQL时间用于箱型图
                if result['predicted_times']:
                    db_difficulty_data[db_name][difficulty].extend(result['predicted_times'])
                
                # 收集每个查询的平均时间用于热力图
                if result['predicted_times']:
                    avg_time = np.mean(result['predicted_times'])
                    db_query_times[db_name].append(avg_time)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 8))
    
    # 子图1：箱型图
    ax1 = plt.subplot(1, 2, 1)
    
    # 准备箱型图数据
    boxplot_data = []
    boxplot_labels = []
    difficulties = ['simple', 'moderate', 'challenging']
    
    for db_name in sorted(db_difficulty_data.keys()):
        for diff in difficulties:
            if diff in db_difficulty_data[db_name] and db_difficulty_data[db_name][diff]:
                boxplot_data.append(db_difficulty_data[db_name][diff])
                boxplot_labels.append(f"{db_name}\n{diff}")
    
    if boxplot_data:
        # box = ax1.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True) ## warning
        box = ax1.boxplot(boxplot_data, patch_artist=True)
        ax1.set_xticklabels(boxplot_labels)
        # 设置颜色
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for i, patch in enumerate(box['boxes']):
            patch.set_facecolor(colors[i % 3])
        
        ax1.set_title('Execution Time by Database and Difficulty', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
    
    # 子图2：热力图
    ax2 = plt.subplot(1, 2, 2)
    
    # 准备热力图数据
    heatmap_data = []
    db_names = sorted(db_query_times.keys())
    
    for db_name in db_names:
        times = db_query_times[db_name]
        if len(times) > 10:  # 如果查询数量多，进行分箱
            hist, bins = np.histogram(times, bins=10)
            heatmap_data.append(hist)
        else:
            # 对于少量查询，直接使用时间值
            padded_times = times + [0] * (10 - len(times))  # 填充到10个
            heatmap_data.append(padded_times[:10])
    
    if heatmap_data and db_names:
        # 转置数据以便更好的可视化
        heatmap_data = np.array(heatmap_data).T
        
        im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        
        # 设置坐标轴
        ax2.set_xticks(range(len(db_names)))
        ax2.set_xticklabels(db_names, rotation=45, ha='right')
        ax2.set_yticks(range(min(10, len(heatmap_data))))
        ax2.set_yticklabels([f'Bin {i+1}' for i in range(min(10, len(heatmap_data)))])
        
        ax2.set_title('Query Time Distribution Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Databases', fontsize=12)
        ax2.set_ylabel('Time Bins', fontsize=12)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax2, label='Frequency')
        
        # 添加数值标注
        for i in range(len(db_names)):
            for j in range(min(10, len(heatmap_data))):
                text = ax2.text(i, j, f'{heatmap_data[j, i]:.0f}',
                               ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'advanced_time_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Advanced time analysis plot saved to: {output_path}")
    print(f"Total databases analyzed: {len(db_names)}")

def plot_time_distribution(exec_result, output_dir):
    """绘制每个数据集的时间分布图"""
    # 按数据库分组数据
    db_data = {}
    for result in exec_result:
        db_name = result['db_name']
        if db_name not in db_data:
            db_data[db_name] = {
                'pred_times': [],
                'gt_times': []
            }
        
        # 添加预测SQL执行时间
        if result['predicted_times']:
            db_data[db_name]['pred_times'].extend(result['predicted_times'])
        
        # 添加真实SQL执行时间
        if result['ground_truth_times']:
            db_data[db_name]['gt_times'].extend(result['ground_truth_times'])
    
    print(f"Found {len(db_data)} databases in results: {list(db_data.keys())}")
    
    # 创建子图 
    n_dbs = len(db_data)
    if n_dbs == 0:
        print("No time data available for plotting")
        return
    
    # 计算子图布局
    cols = min(3, n_dbs)
    rows = (n_dbs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_dbs == 1:
        axes = [axes]
    elif rows > 1 and cols > 1:
        axes = axes.flatten()
    else:
        axes = list(axes)
    
    # 为每个数据库绘制时间分布
    for idx, (db_name, data) in enumerate(db_data.items()):
        if idx < len(axes):
            ax = axes[idx]
            
            # 准备数据
            pred_times = data['pred_times']
            gt_times = data['gt_times']
            
            # 创建直方图
            if pred_times:
                ax.hist(pred_times, bins=20, alpha=0.7, label='Predicted SQL', color='blue', edgecolor='black')
            if gt_times:
                ax.hist(gt_times, bins=20, alpha=0.7, label='Ground Truth SQL', color='red', edgecolor='black')
            
            ax.set_title(f'Database: {db_name}\n(P: {len(pred_times)}, GT: {len(gt_times)})')
            ax.set_xlabel('Execution Time (seconds)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(len(db_data), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'time_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time distribution plot saved to: {output_path}")
    
    # 同时保存时间统计信息
    stats = {}
    for db_name, data in db_data.items():
        pred_times = data['pred_times']
        gt_times = data['gt_times']
        
        stats[db_name] = {
            'predicted_sql': {
                'count': len(pred_times),
                'mean_time': np.mean(pred_times) if pred_times else 0,
                'std_time': np.std(pred_times) if pred_times else 0,
                'min_time': np.min(pred_times) if pred_times else 0,
                'max_time': np.max(pred_times) if pred_times else 0
            },
            'ground_truth_sql': {
                'count': len(gt_times),
                'mean_time': np.mean(gt_times) if gt_times else 0,
                'std_time': np.std(gt_times) if gt_times else 0,
                'min_time': np.min(gt_times) if gt_times else 0,
                'max_time': np.max(gt_times) if gt_times else 0
            }
        }
    
    stats_path = os.path.join(output_dir, 'time_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Time statistics saved to: {stats_path}")

def print_data(score_lists, count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('=========================================    VES   ========================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('ves', *score_lists))

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
    args_parser.add_argument('--diff_json_path', type=str, default='')
    args_parser.add_argument('--iterate_num', type=int, default=100, help='Number of iterations for VES calculation')
    args_parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for plots and stats')
    args = args_parser.parse_args()
    exec_result = []

    # 创建输出目录
    output_dir = os.path.join(args.predicted_sql_path,"vis")
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    
    # 在package_sqls阶段就进行过滤
    pred_queries, db_paths, filtered_sql_keys = package_sqls(
        args.predicted_sql_path, args.ground_truth_path, args.db_root_path, 
        diff_json_path=args.diff_json_path,
        mode=args.mode_predict, 
        data_mode=args.data_mode
    )
    
    # generate gt sqls (使用相同的过滤逻辑)
    gt_queries, db_paths_gt, _ = package_sqls(
        args.predicted_sql_path, args.ground_truth_path, args.db_root_path,
        diff_json_path=args.diff_json_path,
        mode='gt', 
        data_mode=args.data_mode
    )

    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus, 
                     iterate_num=args.iterate_num, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)
    
    # 绘制时间分布图
    plot_time_distribution(exec_result, output_dir)
    plot_advanced_time_analysis(exec_result, args.diff_json_path, filtered_sql_keys, output_dir)
    
    print('start calculate')
    simple_ves, moderate_ves, challenging_ves, ves, count_lists = \
        compute_ves_by_diff(exec_result, args.diff_json_path, args.predicted_sql_path, args.data_mode, filtered_sql_keys)
    score_lists = [simple_ves, moderate_ves, challenging_ves, ves]
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print("Finished evaluation")
    