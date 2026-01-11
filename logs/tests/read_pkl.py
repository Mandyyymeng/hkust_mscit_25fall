import pickle
import glob
import sys
from datetime import datetime
from pathlib import Path

class FileLogger:
    """文件日志输出类"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.original_stdout = sys.stdout
        
    def write(self, text):
        """只输出到文件，不输出到控制台"""
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(text)
    
    def flush(self):
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.flush()

def setup_logging(output_file):
    """设置日志输出到文件"""
    # 确保输出目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 清空或创建文件并写入头部
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"=== MCTS 节点分析日志 ===\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
    
    # 重定向标准输出到文件
    sys.stdout = FileLogger(output_file)

def print_node_details(node, is_root=False):
    """
    简洁地打印节点的详细信息
    """
    if is_root:
        # 根节点显示通用信息
        print("  📋 根节点信息:")
        print(f"    类型: {node.node_type.value}")
        print(f"    深度: {node.depth}")
        
        if hasattr(node, 'original_question') and node.original_question:
            question_preview = node.original_question[:80] + "..." if len(node.original_question) > 80 else node.original_question
            print(f"    问题: {question_preview}")
        
        if hasattr(node, 'hint') and node.hint:
            hint_preview = node.hint[:80] + "..." if len(node.hint) > 80 else node.hint
            print(f"    提示: {hint_preview}")
        
        if hasattr(node, 'db_id'):
            print(f"    数据库: {node.db_id}")
        
        if hasattr(node, 'schema_context') and node.schema_context:
            schema_preview = node.schema_context[:100] + "..." if len(node.schema_context) > 100 else node.schema_context
            print(f"    模式上下文预览: {schema_preview}")
        
        print("    ─────────────────────────")
    else:
        # 非根节点显示具体操作内容
        node_type = node.node_type.value if hasattr(node.node_type, 'value') else node.node_type
        print(f"  📋 {node_type} 节点 (深度: {node.depth}):")
        
        # 显示父动作类型
        if hasattr(node, 'parent_action') and node.parent_action:
            action_type = type(node.parent_action).__name__
            print(f"    父动作: {action_type}")
        
        # 根据节点类型和属性显示具体内容
        if hasattr(node, 'rephrased_question') and node.rephrased_question:
            rephrased_preview = node.rephrased_question[:80] + "..." if len(node.rephrased_question) > 80 else node.rephrased_question
            print(f"    重述问题: {rephrased_preview}")
        
        if hasattr(node, 'selected_schema_context') and node.selected_schema_context:
            selected_schema_preview = node.selected_schema_context
            print(f"    选择模式: {selected_schema_preview}")
        
        if hasattr(node, 'identified_column_values') and node.identified_column_values:
            column_values_preview = node.identified_column_values
            print(f"    识别列值: {column_values_preview}")
        
        if hasattr(node, 'identified_column_functions') and node.identified_column_functions:
            functions_preview = node.identified_column_functions
            print(f"    识别函数: {functions_preview}")
        
        # SQL相关
        if hasattr(node, 'sql_query') and node.sql_query:
            sql_preview = node.sql_query[:80] + "..." if len(node.sql_query) > 80 else node.sql_query
            print(f"    SQL查询: {sql_preview}")
        
        if hasattr(node, 'revised_sql_query') and node.revised_sql_query:
            revised_preview = node.revised_sql_query[:80] + "..." if len(node.revised_sql_query) > 80 else node.revised_sql_query
            print(f"    修订SQL: {revised_preview}")
        
        if hasattr(node, 'final_sql_query') and node.final_sql_query:
            final_preview = node.final_sql_query[:80] + "..." if len(node.final_sql_query) > 80 else node.final_sql_query
            print(f"    最终SQL: {final_preview}")
        
        # 验证和评分
        if hasattr(node, 'is_valid_sql_query') and node.is_valid_sql_query is not None:
            validity = "有效" if node.is_valid_sql_query else "无效"
            print(f"    SQL有效性: {validity}")
        
        if hasattr(node, 'consistency_score') and node.consistency_score is not None:
            print(f"    一致性评分: {node.consistency_score:.3f}")
        
        # 选择的模式信息
        if hasattr(node, 'selected_schema_dict') and node.selected_schema_dict:
            table_count = len(node.selected_schema_dict)
            column_count = sum(len(table.columns) for table in node.selected_schema_dict.values() if hasattr(table, 'columns'))
            print(f"    选择模式: {table_count}表/{column_count}列")
        
        # 路径信息
        if hasattr(node, 'path_nodes') and node.path_nodes:
            print(f"    路径长度: {len(node.path_nodes)}")
        
        # 统计信息（如果有）
        if hasattr(node, 'N') and hasattr(node, 'Q'):
            print(f"    访问次数: {node.N}, 累计奖励: {node.Q:.2f}")
        
        print("    ─────────────────────────")

def print_simple_tree_structure(path, max_depth=5, show_details=False):
    """
    简单打印单个路径的树形结构
    """
    if not path or len(path) == 0:
        print("  路径为空")
        return
    
    print(f"\n🌳 路径树结构 (长度: {len(path)} 个节点):")
    print("=" * 50)
    
    for i, node in enumerate(path):
        level = i
        prefix = "  " * level + "└── " if i > 0 else "🌱 "
        
        # 获取节点基本信息
        node_type = getattr(node, 'node_type', 'Unknown')
        if hasattr(node_type, 'value'):
            node_type = node_type.value
        
        depth = getattr(node, 'depth', i)
        
        # 显示节点信息
        node_info = f"{node_type} (深度: {depth})"
        
        # 如果有SQL查询，显示预览
        if hasattr(node, 'final_sql_query') and node.final_sql_query and i == len(path) - 1:
            sql_preview = node.final_sql_query[:50] + "..." if len(node.final_sql_query) > 50 else node.final_sql_query
            node_info += f" -> SQL: {sql_preview}"
        
        print(f"{prefix}{node_info}")
        
        # 如果需要显示详细信息
        if show_details:
            if i == 0:
                # 根节点显示通用信息
                print_node_details(node, is_root=True)
            else:
                # 其他节点显示具体操作内容
                print_node_details(node, is_root=False)
        
        # 如果达到最大深度，停止打印
        if i >= max_depth - 1 and i < len(path) - 1:
            print("  " * (level + 1) + "└── ... (后续节点省略)")
            break

def view_one_file(file_path, show_node_details=False):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"  类型: {type(data)}")
        print(f"  长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        if isinstance(data, list):
            print(f"  列表包含 {len(data)} 个推理路径")
            if len(data) > 0 and isinstance(data[0], list):
                avg_nodes = sum(len(path) for path in data) / len(data)
                print(f"  每个路径平均节点数: {avg_nodes:.1f}")
                
                # 显示路径长度分布
                path_lengths = [len(path) for path in data]
                from collections import Counter
                length_counts = Counter(path_lengths)
                print(f"  路径长度分布: {dict(length_counts)}")
                
                # 显示前3个路径的节点类型序列
                for i, path in enumerate(data[:3]):
                    print(f"    路径{i+1}: {len(path)} 个节点")
                    node_types = [node.node_type.value if hasattr(node, 'node_type') else type(node).__name__ for node in path]
                    print(f"      节点类型序列: {node_types}")
        
        # 尝试获取question_id（从文件名）
        import re
        match = re.search(r'(\d+)\.pkl', file_path)
        question_id = match.group(1) if match else "unknown"
        if match:
            print(f"  Question ID: {question_id}")
            
        # 显示文件中的SQL查询（如果有）
        if isinstance(data, list) and len(data) > 0:
            print(f"  包含的SQL查询示例:")
            sql_count = 0
            for i, path in enumerate(data):
                if len(path) > 0 and hasattr(path[-1], 'final_sql_query') and path[-1].final_sql_query:
                    sql = path[-1].final_sql_query
                    print(f"    路径{i+1} SQL: {sql}")
                    sql_count += 1
                    if sql_count >= 4:  # 只显示前4个SQL
                        break
        
        # 只打印第一个路径的树结构（避免序列化问题）
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            first_path = data[0]
            print_simple_tree_structure(first_path, show_details=show_node_details)
                
    except Exception as e:
        print(f"  ❌ 读取错误: {e}")
        import traceback
        print(f"  详细错误: {traceback.format_exc()}")

def analyze_and_save_to_file(file_path, output_file, show_node_details=True):
    """
    分析文件并保存到指定输出文件（静默模式，不输出到控制台）
    """
    # 保存原始标准输出
    original_stdout = sys.stdout
    
    try:
        # 设置输出到文件
        setup_logging(output_file)
        
        # 写入分析开始信息
        print(f"开始分析文件: {file_path}")
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 分析文件（所有输出直接到文件）
        view_one_file(file_path, show_node_details=show_node_details)
        
        # 写入分析结束信息
        print("\n" + "=" * 60)
        print("分析完成!")
        
    finally:
        # 恢复标准输出
        sys.stdout = original_stdout

# 使用示例
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # 配置路径
    input_dir = "results/Qwen2.5-Coder-7B-Instruct/kramabench/dev_main"
    output_dir = "logs/dev_main_analysis"
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 查找所有pkl文件
    pkl_files = list(Path(input_dir).glob("*.pkl"))
    
    if not pkl_files:
        print(f"在目录 {input_dir} 中没有找到pkl文件")
        exit(1)
    
    print(f"找到 {len(pkl_files)} 个pkl文件，开始分析...")
    
    # 遍历每个pkl文件
    for pkl_file in pkl_files:
        # 获取文件名（不带扩展名）
        file_stem = pkl_file.stem
        
        # 构建输出文件路径
        output_file = Path(output_dir) / f"{file_stem}.log"
        
        # 分析并保存到文件
        analyze_and_save_to_file(str(pkl_file), str(output_file), show_node_details=True)
        
        print(f"✅ {file_stem}.pkl -> {file_stem}.log")
    
    print(f"\n🎉 所有文件分析完成！")
    print(f"📁 结果保存在: {output_dir}")
    print(f"📊 共处理了 {len(pkl_files)} 个文件")
    
    