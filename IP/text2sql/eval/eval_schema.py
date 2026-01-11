import json
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(predict_file, tables_file, questions_file):
    """加载所有必要的数据文件"""
    with open(predict_file, 'r') as f:
        predict_data = json.load(f)
    with open(tables_file, 'r') as f:
        tables_data = json.load(f)
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    return predict_data, tables_data, questions_data

def parse_query(query):
    """解析查询，分离SQL和数据库ID"""
    if '\t----- bird -----\t' in query:
        sql_part, db_id = query.split('\t----- bird -----\t')
        return sql_part.strip(), db_id.strip()
    return query, None

def extract_table_column_pairs(sql):
    """提取SQL中的表-列对，处理带表前缀的列名"""
    # 移除字符串常量
    sql_clean = re.sub(r"'[^']*'", "' '", sql)
    
    # 提取所有带表前缀的列名 (table.column)
    table_column_pairs = re.findall(r'([\w]+)\.([\w]+)', sql_clean)
    
    # 提取SELECT子句中的列（可能带表前缀）
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_clean, re.IGNORECASE | re.DOTALL)
    columns_in_select = []
    if select_match:
        select_clause = select_match.group(1)
        # 分割逗号分隔的列
        columns = re.split(r',', select_clause)
        for col in columns:
            # 提取可能的表.列格式
            tc_match = re.search(r'([\w]+)\.([\w]+)', col)
            if tc_match:
                columns_in_select.append((tc_match.group(1), tc_match.group(2)))
    
    all_pairs = set(table_column_pairs + columns_in_select)
    return all_pairs

def extract_tables_from_joins(sql):
    """从JOIN条件中提取表名"""
    sql_clean = re.sub(r"'[^']*'", "' '", sql)
    
    tables = set()
    # FROM 子句中的表
    from_matches = re.findall(r'FROM\s+([\w]+)', sql_clean, re.IGNORECASE)
    tables.update(from_matches)
    
    # JOIN 子句中的表
    join_matches = re.findall(r'JOIN\s+([\w]+)', sql_clean, re.IGNORECASE)
    tables.update(join_matches)
    
    return tables


def analyze_sql_correctness(predict_file, tables_file, questions_file):
    """分析SQL正确性 - 修改为区分同一DB其他表 vs 其他DB的列"""
    predict_data, tables_data, questions_data = load_data(predict_file, tables_file, questions_file)
    
    # 构建所有数据库的schema映射
    all_db_schemas = {}
    for db in tables_data:
        db_id = db['db_id']
        
        # 表结构：表名 -> 列集合
        table_columns = defaultdict(set)
        table_names = set(db['table_names_original'])
        
        # 构建表-列映射
        for i, col_info in enumerate(db['column_names_original']):
            table_idx, col_name = col_info[0], col_info[1]
            if table_idx >= 0 and table_idx < len(table_names):  # 有效表索引
                table_name = db['table_names_original'][table_idx]
                table_columns[table_name].add(col_name)
        
        # 收集所有列名（用于判断凭空捏造的列）
        all_columns = set()
        for columns in table_columns.values():
            all_columns.update(columns)
        
        all_db_schemas[db_id] = {
            'tables': table_names,
            'table_columns': dict(table_columns),
            'all_columns': all_columns
        }
    
    # 构建问题ID到详细信息的映射
    question_map = {}
    for q in questions_data:
        question_map[q['question_id']] = q
    
    # 统计结果
    results = {
        'by_db': defaultdict(lambda: {
            'total_queries': 0,
            'table_correctness': [],
            'column_correctness': [],
            'union_usage': 0,
            'column_error_analysis': defaultdict(int),
            'conditional_analysis': {
                'table_correct_errors': defaultdict(int),
                'table_incorrect_errors': defaultdict(int),
                'total_table_correct_columns': 0,
                'total_table_incorrect_columns': 0
            },
            'details': []
        }),
        'by_difficulty': defaultdict(lambda: {
            'total_queries': 0,
            'table_correctness': [],
            'column_correctness': [],
            'union_usage': 0,
            'column_error_analysis': defaultdict(int),
            'conditional_analysis': {
                'table_correct_errors': defaultdict(int),
                'table_incorrect_errors': defaultdict(int),
                'total_table_correct_columns': 0,
                'total_table_incorrect_columns': 0
            }
        }),
        'overall': {
            'total_queries': 0,
            'table_correctness': [],
            'column_correctness': [],
            'union_usage': 0,
            'column_error_analysis': defaultdict(int),
            'conditional_analysis': {
                'table_correct_errors': defaultdict(int),
                'table_incorrect_errors': defaultdict(int),
                'total_table_correct_columns': 0,
                'total_table_incorrect_columns': 0
            }
        }
    }
    
    # 分析每个查询
    for key, query in predict_data.items():
        sql, db_id = parse_query(query)
        
        if not db_id or db_id not in all_db_schemas:
            continue
        
        # 获取问题信息
        try:
            question_id = int(key)
            question_info = question_map.get(question_id, {})
            difficulty = question_info.get('difficulty', 'unknown')
        except:
            difficulty = 'unknown'
        
        # 获取当前数据库的schema和其他所有数据库的信息
        current_db_schema = all_db_schemas[db_id]
        valid_tables = current_db_schema['tables']
        table_columns = current_db_schema['table_columns']
        current_db_all_columns = current_db_schema['all_columns']
        
        # 提取表名和表-列对
        tables_in_sql = extract_tables_from_joins(sql)
        table_column_pairs = extract_table_column_pairs(sql)
        
        # 计算表正确率
        correct_tables = [t for t in tables_in_sql if t in valid_tables]
        table_correctness = len(correct_tables) / len(tables_in_sql) if tables_in_sql else 1.0
        
        # 判断表是否正确（所有表都正确）
        all_tables_correct = len(correct_tables) == len(tables_in_sql) and len(tables_in_sql) > 0
        
        # 计算列正确率和错误分类
        correct_columns = 0
        total_columns = 0
        column_details = []
        error_analysis = defaultdict(int)
        
        for table, column in table_column_pairs:
            total_columns += 1
            
            # 检查表是否存在
            table_exists = table in valid_tables
            
            if table_exists and column in table_columns[table]:
                # 完全正确的列引用
                correct_columns += 1
                column_details.append(f"✓ {table}.{column}")
            else:
                # 错误的列引用 - 详细分类
                column_exists_in_current_db = column in current_db_all_columns
                
                # 检查列是否存在于其他数据库中
                column_exists_in_other_db = False
                other_dbs_with_column = []
                for other_db_id, other_schema in all_db_schemas.items():
                    if other_db_id != db_id and column in other_schema['all_columns']:
                        column_exists_in_other_db = True
                        other_dbs_with_column.append(other_db_id)
                
                error_type = None
                
                if not table_exists:
                    # 表不存在
                    if column_exists_in_current_db:
                        # 表不存在，但列存在于当前数据库的其他表中
                        error_type = 'column_in_same_db_other_table'
                        actual_tables = [t for t, cols in table_columns.items() if column in cols]
                        column_details.append(f"✗ {table}.{column} (invalid table, column in same DB tables: {actual_tables})")
                    elif column_exists_in_other_db:
                        # 表不存在，列存在于其他数据库中
                        error_type = 'column_in_other_db'
                        column_details.append(f"✗ {table}.{column} (invalid table, column in other DBs: {other_dbs_with_column})")
                    else:
                        # 表不存在，列也不存在任何数据库中
                        error_type = 'invalid_table_fictional_column'
                        column_details.append(f"✗ {table}.{column} (invalid table + fictional column)")
                else:
                    # 表存在，但列错误
                    if column_exists_in_current_db:
                        # 列存在于当前数据库的其他表中
                        error_type = 'column_in_same_db_other_table'
                        actual_tables = [t for t, cols in table_columns.items() if column in cols]
                        column_details.append(f"✗ {table}.{column} (valid table, column in other tables: {actual_tables})")
                    elif column_exists_in_other_db:
                        # 表存在，但列存在于其他数据库中
                        error_type = 'column_in_other_db'
                        column_details.append(f"✗ {table}.{column} (valid table, column in other DBs: {other_dbs_with_column})")
                    else:
                        # 表存在，但列凭空捏造
                        error_type = 'fictional_column_valid_table'
                        column_details.append(f"✗ {table}.{column} (valid table but fictional column)")
                
                # 更新错误统计
                error_analysis[error_type] += 1
                
                # 更新条件频率统计
                if not table_exists:
                    results['overall']['conditional_analysis']['table_incorrect_errors'][error_type] += 1
                    results['by_db'][db_id]['conditional_analysis']['table_incorrect_errors'][error_type] += 1
                    results['by_difficulty'][difficulty]['conditional_analysis']['table_incorrect_errors'][error_type] += 1
                else:
                    results['overall']['conditional_analysis']['table_correct_errors'][error_type] += 1
                    results['by_db'][db_id]['conditional_analysis']['table_correct_errors'][error_type] += 1
                    results['by_difficulty'][difficulty]['conditional_analysis']['table_correct_errors'][error_type] += 1
        
        # 更新条件频率的总列数
        if all_tables_correct:
            results['overall']['conditional_analysis']['total_table_correct_columns'] += total_columns
            results['by_db'][db_id]['conditional_analysis']['total_table_correct_columns'] += total_columns
            results['by_difficulty'][difficulty]['conditional_analysis']['total_table_correct_columns'] += total_columns
        else:
            results['overall']['conditional_analysis']['total_table_incorrect_columns'] += total_columns
            results['by_db'][db_id]['conditional_analysis']['total_table_incorrect_columns'] += total_columns
            results['by_difficulty'][difficulty]['conditional_analysis']['total_table_incorrect_columns'] += total_columns
        
        column_correctness = correct_columns / total_columns if total_columns > 0 else 1.0
        
        # 检查UNION
        has_union = 'UNION' in sql.upper()
        
        # 更新统计
        for category in ['overall', 'by_db', 'by_difficulty']:
            if category == 'overall':
                target = results['overall']
            elif category == 'by_db':
                target = results['by_db'][db_id]
            else:
                target = results['by_difficulty'][difficulty]
            
            target['total_queries'] += 1
            target['table_correctness'].append(table_correctness)
            target['column_correctness'].append(column_correctness)
            if has_union:
                target['union_usage'] += 1
            
            # 累加错误分析
            for error_type, count in error_analysis.items():
                target['column_error_analysis'][error_type] += 1
        
        # 保存详细信息
        results['by_db'][db_id]['details'].append({
            'question_id': key,
            'table_correctness': table_correctness,
            'column_correctness': column_correctness,
            'all_tables_correct': all_tables_correct,
            'column_details': column_details,
            'error_analysis': dict(error_analysis),
            'sql': sql
        })
    
    return results, all_db_schemas


def calculate_statistics(results):
    """计算统计指标 - 修复 conditional_analysis 问题"""
    stats = {}
    
    for category, data in results.items():
        if category == 'overall':
            total = data['total_queries']
            avg_table_correctness = sum(data['table_correctness']) / len(data['table_correctness']) if data['table_correctness'] else 0
            avg_column_correctness = sum(data['column_correctness']) / len(data['column_correctness']) if data['column_correctness'] else 0
            union_pct = (data['union_usage'] / total * 100) if total > 0 else 0
            
            stats[category] = {
                'avg_table_correctness': avg_table_correctness * 100,
                'avg_column_correctness': avg_column_correctness * 100,
                'union_pct': union_pct,
                'error_analysis': dict(data['column_error_analysis']),
                'conditional_analysis': data.get('conditional_analysis', {})  # 添加这行
            }
        else:
            stats[category] = {}
            for key, values in data.items():
                total = values['total_queries']
                avg_table_correctness = sum(values['table_correctness']) / len(values['table_correctness']) if values['table_correctness'] else 0
                avg_column_correctness = sum(values['column_correctness']) / len(values['column_correctness']) if values['column_correctness'] else 0
                union_pct = (values['union_usage'] / total * 100) if total > 0 else 0
                
                stats[category][key] = {
                    'avg_table_correctness': avg_table_correctness * 100,
                    'avg_column_correctness': avg_column_correctness * 100,
                    'union_pct': union_pct,
                    'error_analysis': dict(values['column_error_analysis']),
                    'conditional_analysis': values.get('conditional_analysis', {})  # 添加这行
                }
    
    return stats

def visualize_results(stats, output_dir):
    """可视化结果"""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 设置全局字体大小和加粗
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.title_fontsize': 13
    })
    
    # 按数据库的统计
    db_data = stats['by_db']
    if db_data:
        dbs = list(db_data.keys())
        table_correctness = [db_data[db]['avg_table_correctness'] for db in dbs]
        column_correctness = [db_data[db]['avg_column_correctness'] for db in dbs]
        union_usage = [db_data[db]['union_pct'] for db in dbs]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
        
        # 图1：表和列正确率 - 调整颜色
        x = range(len(dbs))
        width = 0.35
        ax1.bar([i - width/2 for i in x], table_correctness, width, 
                label='Table Correctness', alpha=0.9, color='#2E86AB', edgecolor='black', linewidth=1.2)
        ax1.bar([i + width/2 for i in x], column_correctness, width, 
                label='Column Correctness', alpha=0.9, color='#A23B72', edgecolor='black', linewidth=1.2)
        ax1.set_xlabel('Database', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Correctness (%)', fontsize=16, fontweight='bold')
        ax1.set_title('SQL Schema Correctness by Database', fontsize=18, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(dbs, rotation=45, ha='right', fontweight='bold')
        ax1.legend(fontsize=14, framealpha=0.9, shadow=True)
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.set_ylim(0, 100)
        
        # 添加数值标签 - 加粗字体
        for i, (tc, cc) in enumerate(zip(table_correctness, column_correctness)):
            ax1.text(i - width/2, tc + 1, f'{tc:.1f}%', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color='darkblue')
            ax1.text(i + width/2, cc + 1, f'{cc:.1f}%', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color='darkred')
        
        # 图2：UNION使用率 - 调整颜色
        ax2.bar(dbs, union_usage, color='#F18F01', alpha=0.9, edgecolor='black', linewidth=1.2)
        ax2.set_xlabel('Database', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Usage (%)', fontsize=16, fontweight='bold')
        ax2.set_title('UNION Usage by Database', fontsize=18, fontweight='bold', pad=20)
        ax2.set_xticklabels(dbs, rotation=45, ha='right', fontweight='bold')
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.set_ylim(0, max(union_usage) + 10 if union_usage else 20)
        
        # 添加数值标签 - 加粗字体
        for i, u in enumerate(union_usage):
            ax2.text(i, u + 1, f'{u:.1f}%', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold', color='darkorange')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sql_correctness_by_database.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 图3：错误类型分布 - 调整颜色
        fig, ax = plt.subplots(figsize=(14, 10))
        error_types = ['invalid_table', 'column_in_other_table', 'fictional_column', 'other_error']
        error_labels = ['Invalid Table', 'Column in Other Table', 'Fictional Column', 'Other Error']
        
        # 更鲜明的颜色方案
        colors = ['#E74C3C', '#3498DB', '#F39C12', '#27AE60']  # 红, 蓝, 橙, 绿
        
        error_data = []
        for db in dbs:
            db_errors = db_data[db]['error_analysis']
            total_errors = sum(db_errors.values())
            if total_errors > 0:
                error_percentages = [db_errors.get(et, 0) / total_errors * 100 for et in error_types]
            else:
                error_percentages = [0] * len(error_types)
            error_data.append(error_percentages)
        
        # 堆叠柱状图
        bottom = [0] * len(dbs)
        
        for i, error_label in enumerate(error_labels):
            values = [error_data[j][i] for j in range(len(dbs))]
            ax.bar(dbs, values, bottom=bottom, label=error_label, 
                   color=colors[i], alpha=0.9, edgecolor='black', linewidth=1.2)
            bottom = [bottom[j] + values[j] for j in range(len(dbs))]
        
        ax.set_xlabel('Database', fontsize=16, fontweight='bold')
        ax.set_ylabel('Error Distribution (%)', fontsize=16, fontweight='bold')
        ax.set_title('Column Error Type Distribution by Database', fontsize=18, fontweight='bold', pad=20)
        ax.set_xticklabels(dbs, rotation=45, ha='right', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, framealpha=0.9, shadow=True)
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.set_ylim(0, 100)
        
        # 添加总错误数标签
        for i, db in enumerate(dbs):
            total_errors = sum(db_data[db]['error_analysis'].values())
            if total_errors > 0:
                ax.text(i, 102, f'Total: {total_errors}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold', color='black')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/column_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

def visualize_conditional_analysis(stats, output_dir):
    """可视化条件频率分析 - 更新为新的错误分类"""
    plt.style.use('default')
    
    # 设置全局字体
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # 总体条件分析
    overall_stats = stats.get('overall', {})
    conditional = overall_stats.get('conditional_analysis', {})
    
    if not conditional:
        print("Warning: No conditional analysis data found!")
        return
    
    # 准备数据 - 更新错误类型
    error_types = ['column_in_same_db_other_table', 'column_in_other_db', 
                   'invalid_table_fictional_column', 'fictional_column_valid_table']
    error_labels = ['Same DB,\nOther Table', 'Other DB', 
                    'Invalid Table,\nFictional Column', 'Valid Table,\nFictional Column']
    
    # 表正确和表错误时的错误分布
    table_correct_errors = conditional.get('table_correct_errors', {})
    table_incorrect_errors = conditional.get('table_incorrect_errors', {})
    
    table_correct_counts = [table_correct_errors.get(et, 0) for et in error_types]
    table_incorrect_counts = [table_incorrect_errors.get(et, 0) for et in error_types]
    
    # 计算百分比
    total_table_correct_errors = sum(table_correct_counts)
    total_table_incorrect_errors = sum(table_incorrect_counts)
    
    if total_table_correct_errors == 0 and total_table_incorrect_errors == 0:
        print("Warning: No error data found for conditional analysis!")
        return
    
    table_correct_pct = [err/total_table_correct_errors*100 if total_table_correct_errors > 0 else 0 
                        for err in table_correct_counts]
    table_incorrect_pct = [err/total_table_incorrect_errors*100 if total_table_incorrect_errors > 0 else 0 
                          for err in table_incorrect_counts]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 颜色方案
    colors = ['#3498DB', '#E74C3C', '#F39C12', '#27AE60']  # 调整颜色顺序
    
    # 图1：表正确时的列错误分布
    if total_table_correct_errors > 0:
        wedges1, texts1, autotexts1 = ax1.pie(table_correct_pct, labels=error_labels, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
        ax1.set_title('Column Errors When Tables Are CORRECT\n(Total Errors: {})'.format(total_table_correct_errors), 
                     fontsize=16, fontweight='bold', pad=20)
        
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        for text in texts1:
            text.set_fontweight('bold')
    else:
        ax1.text(0.5, 0.5, 'No Errors\nWhen Tables Are Correct', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax1.set_title('Column Errors When Tables Are CORRECT', fontsize=16, fontweight='bold', pad=20)
    
    # 图2：表错误时的列错误分布
    if total_table_incorrect_errors > 0:
        wedges2, texts2, autotexts2 = ax2.pie(table_incorrect_pct, labels=error_labels, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
        ax2.set_title('Column Errors When Tables Are INCORRECT\n(Total Errors: {})'.format(total_table_incorrect_errors), 
                     fontsize=16, fontweight='bold', pad=20)
        
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        for text in texts2:
            text.set_fontweight('bold')
    else:
        ax2.text(0.5, 0.5, 'No Errors\nWhen Tables Are Incorrect', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax2.set_title('Column Errors When Tables Are INCORRECT', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/conditional_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 图3：堆叠柱状图对比
    if total_table_correct_errors > 0 or total_table_incorrect_errors > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(error_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, table_correct_pct, width, label='Tables Correct', 
                       color='#2E86AB', alpha=0.9, edgecolor='black')
        bars2 = ax.bar(x + width/2, table_incorrect_pct, width, label='Tables Incorrect', 
                       color='#A23B72', alpha=0.9, edgecolor='black')
        
        ax.set_xlabel('Error Type', fontsize=16, fontweight='bold')
        ax.set_ylabel('Error Distribution (%)', fontsize=16, fontweight='bold')
        ax.set_title('Column Error Distribution by Table Correctness Condition', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(error_labels, rotation=45, ha='right', fontweight='bold')
        ax.legend(fontsize=14, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (pct1, pct2) in enumerate(zip(table_correct_pct, table_incorrect_pct)):
            if pct1 > 0:
                ax.text(i - width/2, pct1 + 1, f'{pct1:.1f}%', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
            if pct2 > 0:
                ax.text(i + width/2, pct2 + 1, f'{pct2:.1f}%', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/conditional_error_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def save_results_to_txt(results, stats, output_file):
    """保存结果到文本文件 - 添加条件分析"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("SQL Schema Correctness Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        # 总体统计
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 40 + "\n")
        overall = stats['overall']
        f.write(f"Total SQL queries analyzed: {results['overall']['total_queries']}\n")
        f.write(f"Average table correctness: {overall['avg_table_correctness']:.2f}%\n")
        f.write(f"Average column correctness: {overall['avg_column_correctness']:.2f}%\n")
        f.write(f"UNION usage: {overall['union_pct']:.2f}%\n\n")
        
        # 条件频率分析
        f.write("CONDITIONAL ERROR ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        conditional = overall['conditional_analysis']
        
        # 表正确时的错误分布
        total_correct_errors = sum(conditional['table_correct_errors'].values())
        f.write(f"When Tables Are CORRECT:\n")
        f.write(f"  Total column errors: {total_correct_errors}\n")
        if total_correct_errors > 0:
            for error_type, count in conditional['table_correct_errors'].items():
                pct = count / total_correct_errors * 100
                error_desc = {
                    'column_in_other_table': 'Column in other table',
                    'fictional_column_valid_table': 'Fictional column'
                }.get(error_type, error_type)
                f.write(f"  - {error_desc}: {count} errors ({pct:.1f}%)\n")
        
        # 表错误时的错误分布
        total_incorrect_errors = sum(conditional['table_incorrect_errors'].values())
        f.write(f"\nWhen Tables Are INCORRECT:\n")
        f.write(f"  Total column errors: {total_incorrect_errors}\n")
        if total_incorrect_errors > 0:
            for error_type, count in conditional['table_incorrect_errors'].items():
                pct = count / total_incorrect_errors * 100
                error_desc = {
                    'column_in_other_table_wrong_table': 'Column in other table',
                    'invalid_table_fictional_column': 'Fictional column'
                }.get(error_type, error_type)
                f.write(f"  - {error_desc}: {count} errors ({pct:.1f}%)\n")
        
        f.write("\n")
        
        # 按数据库统计
        f.write("STATISTICS BY DATABASE:\n")
        f.write("=" * 40 + "\n")
        for db_id, db_stats in stats['by_db'].items():
            db_results = results['by_db'][db_id]
            f.write(f"\nDatabase: {db_id}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total queries: {db_results['total_queries']}\n")
            f.write(f"Table correctness: {db_stats['avg_table_correctness']:.2f}%\n")
            f.write(f"Column correctness: {db_stats['avg_column_correctness']:.2f}%\n")
            f.write(f"UNION usage: {db_stats['union_pct']:.2f}%\n")
            
            # 数据库错误分析
            total_db_errors = sum(db_stats['error_analysis'].values())
            if total_db_errors > 0:
                f.write(f"\nColumn Error Analysis ({total_db_errors} total errors):\n")
                for error_type, count in db_stats['error_analysis'].items():
                    percentage = (count / total_db_errors * 100) if total_db_errors > 0 else 0
                    error_desc = {
                        'invalid_table': 'Invalid table reference',
                        'column_in_other_table': 'Column exists in other table',
                        'fictional_column': 'Fictional column (not in any table)',
                        'other_error': 'Other errors'
                    }.get(error_type, error_type)
                    f.write(f"  - {error_desc}: {count} errors ({percentage:.1f}%)\n")
            
            # 显示问题最多的几个查询
            problematic_queries = [d for d in db_results['details'] if d['column_correctness'] < 0.5]
            if problematic_queries:
                f.write(f"\nSample Problematic Queries:\n")
                for pq in problematic_queries[:3]:  # 显示前3个
                    f.write(f"  - Question {pq['question_id']}:\n")
                    f.write(f"    Column correctness: {pq['column_correctness']*100:.1f}%\n")
                    f.write(f"    Error types: {pq['error_analysis']}\n")
                    f.write(f"    SQL: {pq['sql'][:100]}...\n")
                    f.write(f"    Column details (first 5):\n")
                    for detail in pq['column_details'][:5]:
                        f.write(f"      {detail}\n")
        
        # 按难度统计
        f.write("\n" + "=" * 60 + "\n")
        f.write("STATISTICS BY DIFFICULTY:\n")
        f.write("=" * 40 + "\n")
        for difficulty, diff_stats in stats['by_difficulty'].items():
            if difficulty != 'unknown':
                diff_results = results['by_difficulty'][difficulty]
                f.write(f"\nDifficulty: {difficulty}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total queries: {diff_results['total_queries']}\n")
                f.write(f"Table correctness: {diff_stats['avg_table_correctness']:.2f}%\n")
                f.write(f"Column correctness: {diff_stats['avg_column_correctness']:.2f}%\n")
                f.write(f"UNION usage: {diff_stats['union_pct']:.2f}%\n")
                
                # 难度级别的错误分析
                total_diff_errors = sum(diff_stats['error_analysis'].values())
                if total_diff_errors > 0:
                    f.write(f"\nError Analysis:\n")
                    for error_type, count in diff_stats['error_analysis'].items():
                        percentage = (count / total_diff_errors * 100) if total_diff_errors > 0 else 0
                        error_desc = {
                            'invalid_table': 'Invalid table',
                            'column_in_other_table': 'Column in other table',
                            'fictional_column': 'Fictional column',
                            'other_error': 'Other errors'
                        }.get(error_type, error_type)
                        f.write(f"  - {error_desc}: {count} errors ({percentage:.1f}%)\n")

def main():
    """主函数"""
    PRED_DIR = "exp_result/bird_dev_300"
    PREDICT_FILE = f"{PRED_DIR}/predict_dev.json"
    TABLES_FILE = "../Alpha-SQL/data/bird/dev/dev_tables.json"
    QUESTIONS_FILE = "../Alpha-SQL/data/bird/dev/dev_all.json"
    OUTPUT_DIR = f"{PRED_DIR}/vis"
    OUTPUT_TXT = f"{OUTPUT_DIR}/sql_correctness_analysis.txt"
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting SQL correctness analysis...")
    
    # 执行分析
    results, db_schemas = analyze_sql_correctness(PREDICT_FILE, TABLES_FILE, QUESTIONS_FILE)
    
    # 计算统计
    stats = calculate_statistics(results)
    
    # 可视化结果
    print("Generating visualizations...")
    visualize_results(stats, OUTPUT_DIR)
    visualize_conditional_analysis(stats, OUTPUT_DIR)  # 新增条件分析可视化
    
    # 保存文本结果
    print("Saving results to file...")
    save_results_to_txt(results, stats, OUTPUT_TXT)
    
    print("Analysis completed!")
    
if __name__ == "__main__":
    main()