import json
import os

def convert_sql_to_predict_format_advanced(sql_json_path, dev_json_path, output_path="data/predict_dev.json",
                                           dataset_name="bird"):
    """
    高级版本的转换函数，处理各种可能的匹配问题
    """
    # 读取sql.json文件
    with open(sql_json_path, 'r', encoding='utf-8') as f:
        sql_data = json.load(f)

    # 读取dev.json文件
    with open(dev_json_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    # 创建多种映射关系
    question_to_db = {}  # question_id -> db_id
    index_to_question = {}  # 索引 -> question_id

    # 构建映射关系
    for i, item in enumerate(dev_data):
        question_id = item.get("question_id")
        db_id = item.get("db_id")

        if question_id is not None and db_id is not None:
            question_to_db[str(question_id)] = db_id
            # 同时建立索引映射（如果sql.json的key是索引而不是question_id）
            index_to_question[str(i)] = str(question_id)
            index_to_question[str(i + 1)] = str(question_id)  # 1-based索引

    # 转换格式
    predict_data = {}
    matched_count = 0

    # 按key排序处理sql.json中的SQL语句
    for key in sorted(sql_data.keys(), key=int):
        sql = sql_data[key]

        # 确保SQL以分号结尾
        if sql and not sql.endswith(';'):
            sql += ';'

        # 尝试多种方式获取数据库ID
        db_id = "unknown_db"

        # 方式1: 直接使用key作为question_id
        if key in question_to_db:
            db_id = question_to_db[key]
            matched_count += 1
        # 方式2: 使用key作为索引
        elif key in index_to_question:
            question_id = index_to_question[key]
            db_id = question_to_db.get(question_id, "unknown_db")
            if db_id != "unknown_db":
                matched_count += 1
        # 方式3: 如果key是数字，尝试直接使用dev_data中的对应位置
        else:
            try:
                idx = int(key) - 1  # 转换为0-based索引
                if 0 <= idx < len(dev_data):
                    db_id = dev_data[idx].get("db_id", "unknown_db")
                    if db_id != "unknown_db":
                        matched_count += 1
            except ValueError:
                pass

        # 构建predict_dev.json格式
        formatted_sql = f"{sql}\t----- {dataset_name} -----\t{db_id}"
        predict_data[key] = formatted_sql

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predict_data, f, indent=4, ensure_ascii=False)

    print(f"转换完成！")
    print(f"SQL文件: {sql_json_path} (共 {len(sql_data)} 条SQL)")
    print(f"Dev文件: {dev_json_path} (共 {len(dev_data)} 个问题)")
    print(f"输出文件: {output_path}")
    print(f"成功匹配数据库ID的语句: {matched_count}/{len(predict_data)}")
    print(f"匹配率: {matched_count / len(predict_data) * 100:.1f}%")


# 使用示例
if __name__ == "__main__":
    # sql_json_path = "../RSL-SQL/src/sql_log_converted/qwen3_bird.json"  
    sql_json_path = "../Alpha-SQL/logs/pred_sqls/pred_sqls_qwen32b_spidertest.json"   # "data/final_sql.json" 
    dev_json_path = "../Alpha-SQL/data/spider/dev.json"  #bird/dev/dev_all.json"
    output_dir = "exp_result/spider_test_qwen32b"
    
    os.makedirs(output_dir,exist_ok=True)
    output_path = f"{output_dir}/predict_dev.json"

    convert_sql_to_predict_format_advanced(sql_json_path, dev_json_path, output_path)
    