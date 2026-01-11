import sys
sys.path.append('/ssddata/zzhangle/Alpha-SQL') 

import pickle
import json
from pathlib import Path
import glob
import os

def extract_sql_from_pkl(pkl_file_path):
    """
    从pkl文件中提取所有路径的最终SQL
    """
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        sql_list = []
        
        if isinstance(data, list):
            for i, path in enumerate(data):
                if isinstance(path, list) and len(path) > 0:
                    # 获取最后一个节点的final_sql_query
                    last_node = path[-1]
                    if hasattr(last_node, 'final_sql_query') and last_node.final_sql_query:
                        sql_list.append(last_node.final_sql_query)
        
        return sql_list
        
    except Exception as e:
        print(f"❌ 读取文件 {pkl_file_path} 时出错: {e}")
        return []

def extract_all_sql_from_directory(input_dir, output_file):
    """
    从目录中所有pkl文件提取SQL并保存
    """
    # 查找所有pkl文件
    pkl_files = list(Path(input_dir).glob("*.pkl"))
    
    if not pkl_files:
        print(f"在目录 {input_dir} 中没有找到pkl文件")
        return {}
    
    print(f"找到 {len(pkl_files)} 个pkl文件，开始提取SQL...")
    
    sql_dict = {}
    
    for pkl_file in pkl_files:
        # 获取文件名（不带扩展名）
        file_stem = pkl_file.stem
        
        # 提取SQL
        sql_list = extract_sql_from_pkl(str(pkl_file))
        sql_dict[file_stem] = sql_list
        
        print(f"✅ {file_stem}.pkl: 提取到 {len(sql_list)} 个SQL")
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sql_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 SQL提取完成！")
    print(f"📁 结果保存在: {output_file}")
    print(f"📊 共处理了 {len(pkl_files)} 个文件")
    
    # 显示统计信息
    total_sql = sum(len(sql_list) for sql_list in sql_dict.values())
    print(f"📝 总共提取了 {total_sql} 个SQL查询")
    
    # 显示每个文件的SQL数量
    print(f"\n📋 各文件SQL数量统计:")
    for file_stem, sql_list in sorted(sql_dict.items()):
        print(f"  {file_stem}: {len(sql_list)} 个SQL")
    
    return sql_dict

def extract_sql_with_details(input_dir, output_file):
    """
    提取SQL并包含更多详细信息
    """
    pkl_files = list(Path(input_dir).glob("*.pkl"))
    
    if not pkl_files:
        print(f"在目录 {input_dir} 中没有找到pkl文件")
        return {}
    
    detailed_dict = {}
    
    for pkl_file in pkl_files:
        file_stem = pkl_file.stem
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            sql_details = []
            
            if isinstance(data, list):
                for path_index, path in enumerate(data):
                    if isinstance(path, list) and len(path) > 0:
                        last_node = path[-1]
                        if hasattr(last_node, 'final_sql_query') and last_node.final_sql_query:
                            sql_details.append({
                                "path_index": path_index,
                                "sql": last_node.final_sql_query,
                                "path_length": len(path),
                                "node_type": last_node.node_type.value if hasattr(last_node, 'node_type') else "Unknown"
                            })
            
            detailed_dict[file_stem] = sql_details
            print(f"✅ {file_stem}.pkl: {len(sql_details)} 个SQL")
            
        except Exception as e:
            print(f"❌ 读取文件 {pkl_file} 时出错: {e}")
            detailed_dict[file_stem] = []
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_dict, f, indent=2, ensure_ascii=False)
    
    return detailed_dict

# 使用示例
if __name__ == "__main__":
    # 配置路径
    input_dir = "results/Qwen2.5-Coder-32B-Instruct/bird/dev_300"
    out_dir = "logs/analysis/bird_300"
    output_file = f"{out_dir}/extracted_sql.json"
    os.makedirs(out_dir,exist_ok=True)
    
    # 方法1: 简单提取（只保存SQL列表）
    print("方法1: 简单提取SQL")
    print("=" * 40)
    sql_dict = extract_all_sql_from_directory(input_dir, output_file)
    
    print("\n" + "=" * 50)
    
    # 方法2: 详细提取（包含路径信息）
    print("方法2: 详细提取SQL")
    print("=" * 40)
    detailed_output_file = f"{out_dir}/extracted_sql_detailed.json"
    detailed_dict = extract_sql_with_details(input_dir, detailed_output_file)
    
    # 显示示例输出结构
    if sql_dict:
        first_key = list(sql_dict.keys())[0]
        print(f"\n📝 输出文件结构示例:")
        print(f"文件: {output_file}")
        print(f"键: {first_key}")
        print(f"值: {len(sql_dict[first_key])} 个SQL的列表")
        
        if sql_dict[first_key]:
            print(f"第一个SQL示例: {sql_dict[first_key][0][:100]}...")