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
matplotlib.use('Agg')  # 使用非交互式后端

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
    result = {
        'sql_idx': idx, 
        'time_ratio': time_ratio,
        'predicted_times': pred_times,
        'ground_truth_times': gt_times,
        'db_name': db_place.split('/')[-2]  # 从路径中提取数据库名
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
    os.makedirs(args.output_dir, exist_ok=True)

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
    plot_time_distribution(exec_result, args.output_dir)
    
    print('start calculate')
    simple_ves, moderate_ves, challenging_ves, ves, count_lists = \
        compute_ves_by_diff(exec_result, args.diff_json_path, args.predicted_sql_path, args.data_mode, filtered_sql_keys)
    score_lists = [simple_ves, moderate_ves, challenging_ves, ves]
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print("Finished evaluation")
    