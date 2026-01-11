import json


def convert_sql_to_files(input_file, sql_output_file, json_output_file):
    # 读取SQL文件
    with open(input_file, 'r', encoding='utf-8') as f:
        sql_lines = [line.strip() for line in f if line.strip()]

    # 确保每条SQL以分号结尾
    processed_sql_lines = []
    for sql in sql_lines:
        if not sql.endswith(';'):
            sql += ';'
        processed_sql_lines.append(sql)

    # 创建JSON数据
    json_data = {str(i): sql for i, sql in enumerate(processed_sql_lines)}  # (,1)

    # 写入SQL文件（带分号）
    with open(sql_output_file, 'w', encoding='utf-8') as f:
        for sql in processed_sql_lines:
            f.write(sql + '\n')

    # 写入JSON文件
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"转换完成！共处理 {len(json_data)} 条SQL语句")


# 使用示例
if __name__ == "__main__":
    input_file = '/ssddata/zzhanglc/RSL-SQL/src/sql_log/bird_qwen3/final_sql.txt'  #"data/final_sql.txt"
    sql_output_file = "data/final_sql.sql"
    json_output_file = "../RSL-SQL/src/sql_log_converted/qwen3_bird.json" # "data/final_sql.json"

    convert_sql_to_files(input_file, sql_output_file, json_output_file)
