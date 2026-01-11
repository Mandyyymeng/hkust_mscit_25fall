import json
import sqlite3
import os
import logging

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def execute_sql_and_compare(db_path, pred_sql, ground_truth_sql):
    """执行预测SQL和真实SQL，比较结果是否一致"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 执行预测SQL
        cursor.execute(pred_sql)
        pred_result = cursor.fetchall()
        
        # 执行真实SQL
        cursor.execute(ground_truth_sql)
        truth_result = cursor.fetchall()
        
        conn.close()
        
        # 比较结果是否完全相同
        return pred_result == truth_result
        
    except Exception as e:
        logging.error(f"SQL执行失败 - 数据库: {os.path.basename(db_path)} - 错误: {e}")
        return False

def generate_acc_json():
    # 读取分析文件
    json_dir = "logs/pred_sqls/pred_sqls_r1"
    with open(f'{json_dir}/birddev-generated_sql_analysis.json', 'r') as f:
        analysis_data = json.load(f)
    
    acc_data = {}
    
    logging.info(f"开始处理 {len(analysis_data)} 个问题")
    
    for i, item in enumerate(analysis_data, 1):
        if i % 100 == 0:
            logging.info(f"处理进度: {i}/{len(analysis_data)}")
        
        question_id = item["question_id"]
        db_file = item["db_file"]
        pred_sql = item["pred_sql"]
        ground_truth = item["ground_truth"]
        
        # 从db_file路径中提取db_id
        # 假设路径格式: data/NL2SQL/BIRD/dev/dev_databases/california_schools/california_schools.sqlite
        db_id = os.path.basename(os.path.dirname(db_file))
        
        # 检查数据库文件是否存在
        if not os.path.exists(db_file):
            logging.warning(f"数据库文件不存在: {db_file}")
            acc_data[str(question_id)] = {
                "db_id": db_id,
                "result": "ERROR: Database not found",
                "if_correct": 0
            }
            continue
        
        # 执行SQL并比较结果
        is_correct = execute_sql_and_compare(db_file, pred_sql, ground_truth)
        
        # 记录结果
        acc_data[str(question_id)] = {
            "db_id": db_id,
            "result": "CORRECT" if is_correct else "INCORRECT",
            "if_correct": 1 if is_correct else 0
        }
        
        if is_correct:
            logging.debug(f"问题 {question_id} 正确")
    
    # 保存acc.json
    with open('acc.json', 'w') as f:
        json.dump(acc_data, f, indent=2)
    
    # 统计正确率
    correct_count = sum(1 for item in acc_data.values() if item["if_correct"] == 1)
    total_count = len(acc_data)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    logging.info("=" * 50)
    logging.info(f"处理完成!")
    logging.info(f"总问题数: {total_count}")
    logging.info(f"正确数量: {correct_count}")
    logging.info(f"准确率: {accuracy:.4f}")
    logging.info(f"acc.json 已生成")

if __name__ == "__main__":
    generate_acc_json()
    