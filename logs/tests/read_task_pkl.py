import pickle
from pathlib import Path
from collections import defaultdict

def analyze_tasks_pkl(file_path):
    """
    专门分析 tasks.pkl 文件
    """
    print("🔍 分析 tasks.pkl 文件")
    print("=" * 60)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"文件路径: {file_path}")
        print(f"数据类型: {type(data)}")
        print(f"数据长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        if isinstance(data, list):
            print(f"这是一个列表，包含 {len(data)} 个元素")
            
            if len(data) > 0:
                first_item = data[0]
                print(f"\n第一个元素的类型: {type(first_item)}")
                
                # 检查是否是Task对象
                if hasattr(first_item, 'question_id'):
                    print("✅ 确认: 这是 Task 对象列表")
                    
                    # # 显示Task对象的属性
                    # print(f"\nTask对象属性:")
                    # attrs = [attr for attr in dir(first_item) if not attr.startswith('_')]
                    # for attr in attrs:
                    #     try:
                    #         value = getattr(first_item, attr)
                    #         value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    #         print(f"  {attr}: {value_preview}")
                    #     except:
                    #         print(f"  {attr}: <无法获取值>")
                    
                    # 显示任务统计信息
                    print(f"\n📊 任务统计:")
                    print(f"  任务ID范围: {min(task.question_id for task in data)} - {max(task.question_id for task in data)}")
                    
                    # 数据库分布
                    db_distribution = {}
                    for task in data:
                        db_id = task.db_id
                        db_distribution[db_id] = db_distribution.get(db_id, 0) + 1
                    
                    print(f"  数据库分布:")
                    for db_id, count in sorted(db_distribution.items()):
                        print(f"    {db_id}: {count} 个任务")
                    
                    # 显示前3个任务的详细信息
                    print(f"\n📋 前3个任务详情:")
                    for i, task in enumerate(data[:3]):
                        print(f"  任务 {i+1} (ID: {task.question_id}):")
                        print(f"    数据库: {task.db_id}")
                        print(f"    问题: {task.question}")
                        print(f"    证据: {task.evidence}")
                        if hasattr(task, 'sql') and task.sql:
                            print(f"    SQL: {task.sql}")
                        if hasattr(task, 'difficulty') and task.difficulty:
                            print(f"    难度: {task.difficulty}")
                        if hasattr(task, 'schema_context') and task.schema_context:
                            schema_preview = task.schema_context[:100] + "..." if len(task.schema_context) > 100 else task.schema_context
                            print(f"    模式上下文预览: {schema_preview}")
                        print()
                        
                else:
                    print("❌ 这不是 Task 对象")
                    print(f"第一个元素的实际类型: {type(first_item)}")
                    print(f"第一个元素的内容: {first_item}")
                    
        else:
            print(f"数据不是列表，而是: {type(data)}")
            
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")

def analyze_relevant_values_pkl(file_path):
    """
    专门分析 relevant_values_for_all_tasks.pkl 文件
    """
    print("🔍 分析 relevant_values_for_all_tasks.pkl 文件")
    print("=" * 60)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"文件路径: {file_path}")
        print(f"数据类型: {type(data)}")
        print(f"数据长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        if isinstance(data, list):
            print(f"这是一个列表，包含 {len(data)} 个元素")
            
            if len(data) > 0:
                first_item = data[0]
                print(f"\n第一个元素的类型: {type(first_item)}")
                
                if isinstance(first_item, (dict, defaultdict)):
                    print("✅ 确认: 这是 defaultdict 列表")
                    
                    # 显示defaultdict的结构
                    print(f"\ndefaultdict 结构分析:")
                    print(f"  包含 {len(first_item)} 个键值对")
                    
                    if len(first_item) > 0:
                        # 显示前3个键值对
                        print(f"\n📊 前3个表-列的相关值:")
                        for i, ((table_name, column_name), values) in enumerate(list(first_item.items())[:3]):
                            print(f"  {i+1}. {table_name}.{column_name}:")
                            print(f"    值数量: {len(values)}")
                            if values:
                                print(f"    示例值: {values[:5]}")  # 显示前5个值
                                if len(values) > 5:
                                    print(f"    ... 还有 {len(values) - 5} 个值")
                            else:
                                print(f"    无相关值")
                            print()
                    
                    # 统计所有任务的相关值信息
                    print(f"\n📈 所有任务的相关值统计:")
                    total_values = 0
                    tasks_with_values = 0
                    max_values_per_task = 0
                    min_values_per_task = float('inf')
                    
                    for i, relevant_values in enumerate(data):
                        if isinstance(relevant_values, (dict, defaultdict)):
                            task_value_count = sum(len(values) for values in relevant_values.values())
                            total_values += task_value_count
                            if task_value_count > 0:
                                tasks_with_values += 1
                            max_values_per_task = max(max_values_per_task, task_value_count)
                            min_values_per_task = min(min_values_per_task, task_value_count)
                    
                    print(f"  总相关值数量: {total_values}")
                    print(f"  平均每个任务相关值: {total_values/len(data):.1f}")
                    print(f"  包含相关值的任务数: {tasks_with_values}/{len(data)}")
                    print(f"  单个任务最小相关值数: {min_values_per_task}")
                    print(f"  单个任务最大相关值数: {max_values_per_task}")
                    
                    # 显示不同任务的相关值数量分布
                    print(f"\n📋 各任务相关值数量:")
                    for i, relevant_values in enumerate(data[:5]):  # 显示前5个任务
                        task_value_count = sum(len(values) for values in relevant_values.values())
                        print(f"  任务 {i+1}: {task_value_count} 个相关值")
                        
                else:
                    print("❌ 这不是 defaultdict")
                    print(f"第一个元素的实际类型: {type(first_item)}")
                    print(f"第一个元素的内容: {first_item}")
                    
        else:
            print(f"数据不是列表，而是: {type(data)}")
            
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")

# 使用示例
if __name__ == "__main__":
    base_dir = "data/preprocessed/kramabench/dev/dev"
    
    # 分析 tasks.pkl
    tasks_file = Path(base_dir) / "tasks.pkl"
    if tasks_file.exists():
        analyze_tasks_pkl(str(tasks_file))
    else:
        print(f"❌ 文件不存在: {tasks_file}")
    
    print("\n" + "=" * 80 + "\n")
    
    # 分析 relevant_values_for_all_tasks.pkl
    relevant_values_file = Path(base_dir) / "relevant_values_for_all_tasks.pkl"
    if relevant_values_file.exists():
        analyze_relevant_values_pkl(str(relevant_values_file))
    else:
        print(f"❌ 文件不存在: {relevant_values_file}")