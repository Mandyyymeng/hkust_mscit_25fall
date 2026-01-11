import json


def convert_json_to_sql(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sql_statements = []
    for key, value in data.items():
        # 分割字符串，取SQL部分（去掉 ----- bird ----- 和数据库名）
        if value==0:
            sql_part = "SELECT 0;"   ### modified
        else:
            sql_part = value.split('\t----- bird -----\t')[0]
        sql_statements.append(sql_part)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sql in sql_statements:
            f.write(sql + '\n')

    print(f"转换完成！共生成 {len(sql_statements)} 条SQL语句")

# 使用示例
if __name__ == "__main__":
    input_file = "exp_result/turbo_output_kg/predict_dev.json"
    output_file = "exp_result/SuperSQL.sql"
    convert_json_to_sql(input_file, output_file)
