from alphasql.database.sql_execution import cached_execute_sql_with_timeout, is_valid_execution_result
from alphasql.algorithm.selection.utils import measure_sql_execution_time
import pickle
import glob
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcts_runner.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

EXECUTION_TIME_REPEAT = 20

def select_final_sql_query0(results_file_path: str, db_root_dir: str):
    question_id = int(results_file_path.split("/")[-1].split(".")[0])
    with open(results_file_path, "rb") as f:
        results = pickle.load(f)
    db_id = results[0][0].db_id
    db_path = f"{db_root_dir}/{db_id}/{db_id}.sqlite"
    
    # 存储所有路径的详细信息
    all_paths_info = []
    result_groups = defaultdict(list)
    result_groups_with_invalid_result = defaultdict(list)
    
    # 第一遍：执行所有SQL并分组
    for idx, result in tqdm(enumerate(results), desc=f"Processing results for {question_id}"):
        sql_query = result[-1].final_sql_query
        answer = cached_execute_sql_with_timeout(db_path, sql_query)
        
        # 记录每个路径的详细信息
        path_info = {
            "path_index": idx,
            "sql": sql_query,
            "execution_result": {
                "result_type": answer.result_type.value,
                "result": answer.result,  # 保持原始类型，不强制转换
                "error_message": answer.error_message if hasattr(answer, 'error_message') else None
            },
            "is_valid": False,
            "path_length": len(result),
            "consistency_score": 0.0,  # 初始化为0，后面计算
            "execution_time": 0.0      # 初始化为0，后面计算
        }
        
        if answer.result_type.value == "success":
            if is_valid_execution_result(answer):
                path_info["is_valid"] = True
                # 使用可哈希的结果进行分组
                result_key = str(answer.result)  # 直接使用字符串表示
                result_groups[result_key].append(idx)
            # 即使是无效结果也记录分组
            result_key = str(answer.result)
            result_groups_with_invalid_result[result_key].append(idx)
        
        all_paths_info.append(path_info)
    
    # 第二遍：计算每个路径的一致性分数
    total_valid_paths = sum(len(v) for v in result_groups.values())
    total_all_paths = sum(len(v) for v in result_groups_with_invalid_result.values())
    
    # 为有效结果组计算分数
    for result_key, path_indices in result_groups.items():
        sc_score = len(path_indices) / total_valid_paths if total_valid_paths > 0 else 0
        # 测量执行时间（只测第一个路径作为代表）
        if path_indices:
            execution_time = measure_sql_execution_time(db_path, results[path_indices[0]][-1].final_sql_query, repeat=EXECUTION_TIME_REPEAT)
            for path_idx in path_indices:
                all_paths_info[path_idx]["consistency_score"] = sc_score
                all_paths_info[path_idx]["execution_time"] = execution_time
    
    # 为无效结果组计算分数
    for result_key, path_indices in result_groups_with_invalid_result.items():
        sc_score = len(path_indices) / total_all_paths if total_all_paths > 0 else 0
        if path_indices and all_paths_info[path_indices[0]]["execution_time"] == 0:
            execution_time = measure_sql_execution_time(db_path, results[path_indices[0]][-1].final_sql_query, repeat=EXECUTION_TIME_REPEAT)
            for path_idx in path_indices:
                if all_paths_info[path_idx]["consistency_score"] == 0:  # 只设置未设置的
                    all_paths_info[path_idx]["consistency_score"] = sc_score
                    all_paths_info[path_idx]["execution_time"] = execution_time
    
    # === 修改选择逻辑，确保 path_idx 始终有定义 ===
    path_idx = -1  # 默认值
    final_selected_sql_query = "ERROR"  # 默认值
    
    if len(result_groups) == 0:
        # 没有有效结果组
        if len(result_groups_with_invalid_result) > 0:
            path_idx_with_sc_score = []
            for result_key, path_indices in result_groups_with_invalid_result.items():
                sc_score = all_paths_info[path_indices[0]]["consistency_score"]
                execution_time = all_paths_info[path_indices[0]]["execution_time"]
                path_idx_with_sc_score.append((path_indices[0], sc_score, execution_time))
            path_idx_with_sc_score.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            path_idx = path_idx_with_sc_score[0][0]
            final_selected_sql_query = results[path_idx][-1].final_sql_query
        else:
            # 没有任何结果的情况
            path_idx = -1
            final_selected_sql_query = "ERROR"
    else:
        # 有有效结果组
        path_idx_with_sc_score = []
        for result_key, path_indices in result_groups.items():
            sc_score = all_paths_info[path_indices[0]]["consistency_score"]
            execution_time = all_paths_info[path_indices[0]]["execution_time"]
            path_idx_with_sc_score.append((path_indices[0], sc_score, execution_time))
        path_idx_with_sc_score.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        path_idx = path_idx_with_sc_score[0][0]
        final_selected_sql_query = results[path_idx][-1].final_sql_query

    # 现在 path_idx 在所有分支中都有定义
    return {
        "question_id": question_id,
        "sql": final_selected_sql_query,
        "db_id": db_id,
        "selected_path_index": path_idx,  # 直接使用，不需要检查 locals()
        "selection_criteria": {
            "consistency_score": all_paths_info[path_idx]["consistency_score"] if path_idx >= 0 else 0,
            "execution_time": all_paths_info[path_idx]["execution_time"] if path_idx >= 0 else 0
        },
        "statistics": {
            "total_paths": len(results),
            "valid_paths": total_valid_paths,
            "result_groups_count": len(result_groups),
            "invalid_result_groups_count": len(result_groups_with_invalid_result)
        },
        "all_paths_info": all_paths_info
    }
    
def select_final_sql_query(results_file_path: str, db_root_dir: str):
    question_id = int(results_file_path.split("/")[-1].split(".")[0])
    with open(results_file_path, "rb") as f:
        results = pickle.load(f)
    db_id = results[0][0].db_id
    db_path = f"{db_root_dir}/{db_id}/{db_id}.sqlite"
    
    # 存储所有路径的详细信息
    all_paths_info = []
    result_groups = defaultdict(list)
    result_groups_with_invalid_result = defaultdict(list)
    
    # 第一遍：执行所有SQL并分组
    for idx, result in tqdm(enumerate(results), desc=f"Processing results for {question_id}"):
        sql_query = result[-1].final_sql_query
        answer = cached_execute_sql_with_timeout(db_path, sql_query)
        
        # 记录每个路径的详细信息
        path_info = {
            "path_index": idx,
            "sql": sql_query,
            "execution_result": {
                "result_type": answer.result_type.value,
                "result": answer.result,  # 保持原始类型，不强制转换
                "error_message": answer.error_message if hasattr(answer, 'error_message') else None
            },
            "is_valid": False,
            "path_length": len(result),
            "consistency_score": 0.0,  # 初始化为0，后面计算
            "execution_time": 0.0      # 初始化为0，后面计算
        }
        
        if answer.result_type.value == "success":
            if is_valid_execution_result(answer):
                path_info["is_valid"] = True
                # 使用可哈希的结果进行分组
                result_key = str(answer.result)  # 直接使用字符串表示
                result_groups[result_key].append(idx)
            # 即使是无效结果也记录分组
            result_key = str(answer.result)
            result_groups_with_invalid_result[result_key].append(idx)
        
        all_paths_info.append(path_info)
    
    # 第二遍：计算每个路径的一致性分数
    total_valid_paths = sum(len(v) for v in result_groups.values())
    total_all_paths = sum(len(v) for v in result_groups_with_invalid_result.values())
    
    # 为有效结果组计算分数
    for result_key, path_indices in result_groups.items():
        sc_score = len(path_indices) / total_valid_paths if total_valid_paths > 0 else 0
        # 测量执行时间（只测第一个路径作为代表）
        if path_indices:
            execution_time = measure_sql_execution_time(db_path, results[path_indices[0]][-1].final_sql_query, repeat=EXECUTION_TIME_REPEAT)
            for path_idx in path_indices:
                all_paths_info[path_idx]["consistency_score"] = sc_score
                all_paths_info[path_idx]["execution_time"] = execution_time
    
    # 为无效结果组计算分数
    for result_key, path_indices in result_groups_with_invalid_result.items():
        sc_score = len(path_indices) / total_all_paths if total_all_paths > 0 else 0
        if path_indices and all_paths_info[path_indices[0]]["execution_time"] == 0:
            execution_time = measure_sql_execution_time(db_path, results[path_indices[0]][-1].final_sql_query, repeat=EXECUTION_TIME_REPEAT)
            for path_idx in path_indices:
                if all_paths_info[path_idx]["consistency_score"] == 0:  # 只设置未设置的
                    all_paths_info[path_idx]["consistency_score"] = sc_score
                    all_paths_info[path_idx]["execution_time"] = execution_time
    
    # === 保持原始的选择逻辑不变 ===
    if len(result_groups) == 0:
        final_selected_sql_query = "ERROR"
        
        if len(result_groups_with_invalid_result) > 0:
            path_idx_with_sc_score = []
            for result_key, path_indices in result_groups_with_invalid_result.items():
                # 使用每个路径自己的一致性分数
                sc_score = all_paths_info[path_indices[0]]["consistency_score"]
                execution_time = all_paths_info[path_indices[0]]["execution_time"]
                path_idx_with_sc_score.append((path_indices[0], sc_score, execution_time))
            path_idx_with_sc_score.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            path_idx = path_idx_with_sc_score[0][0]
            final_selected_sql_query = results[path_idx][-1].final_sql_query
    else:
        path_idx_with_sc_score = []
        for result_key, path_indices in result_groups.items():
            # 使用每个路径自己的一致性分数
            sc_score = all_paths_info[path_indices[0]]["consistency_score"]
            execution_time = all_paths_info[path_indices[0]]["execution_time"]
            path_idx_with_sc_score.append((path_indices[0], sc_score, execution_time))
        path_idx_with_sc_score.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        path_idx = path_idx_with_sc_score[0][0]
        final_selected_sql_query = results[path_idx][-1].final_sql_query

    # 在返回结果中添加详细信息（不影响原有逻辑）
    return {
        "question_id": question_id,
        "sql": final_selected_sql_query,  # 保持原有字段名
        # 新增详细信息
        "db_id": db_id,
        "selected_path_index": path_idx if 'path_idx' in locals() else -1,
        "selection_criteria": {
            "consistency_score": all_paths_info[path_idx]["consistency_score"] if path_idx >= 0 else 0,
            "execution_time": all_paths_info[path_idx]["execution_time"] if path_idx >= 0 else 0
        },
        "statistics": {
            "total_paths": len(results),
            "valid_paths": total_valid_paths,
            "result_groups_count": len(result_groups),
            "invalid_result_groups_count": len(result_groups_with_invalid_result)
        },
        "all_paths_info": all_paths_info  # 保存所有路径的详细信息，每个都有consistency_score
    }

def main(args):
    final_pred_sqls = {}
    detailed_results = {}
    
    with ProcessPoolExecutor(max_workers=args.process_num) as executor:
        result_paths = glob.glob(args.results_dir + "/*.pkl")
        future_to_path = {executor.submit(select_final_sql_query, path, args.db_root_dir): path for path in result_paths}
        
        for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Processing results"):
            result = future.result()
            question_id = str(result["question_id"])
            
            # 保持原有格式（只包含sql字段）
            final_pred_sqls[question_id] = result["sql"]
            
            # 保存详细结果
            detailed_results[question_id] = result
    
    # 保存原有格式的JSON（保持向后兼容）
    with open(args.output_path, "w") as f:
        json.dump(final_pred_sqls, f, indent=4)
    logger.info(f"Results saved to: {args.output_path}")
    
    # 可选：保存详细结果到另一个文件
    if True: #args.save_detailed:
        detailed_output_path = args.output_path.replace(".json", "_detailed.json")
        with open(detailed_output_path, "w") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to: {detailed_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--db_root_dir", type=str, required=True)
    parser.add_argument("--process_num", type=int, default=32)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--save_detailed", action="store_true", help="Save detailed results to separate file")
    args = parser.parse_args()
    main(args)