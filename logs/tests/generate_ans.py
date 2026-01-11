import json
import sqlite3
import os
import logging

# 配置logging（仅控制台）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def generate_answers():
    base_dir = "data/bird/dev"
    with open(f'{base_dir}/dev_all.json', 'r') as f:
        dev_data = json.load(f)
    
    answers = {}
    
    logging.info(f"Starting to process {len(dev_data)} questions")
    
    for i, item in enumerate(dev_data, 1):
        if i % 100 == 0:
            logging.info(f"Processed {i}/{len(dev_data)} questions")
        
        db_path = f"{base_dir}/dev_databases/{item['db_id']}/{item['db_id']}.sqlite"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(item["SQL"])
        result = cursor.fetchone()[0]
        conn.close()
        
        answers[item["question_id"]] = str(result)
    
    logging.info(f"Completed! Processed all {len(dev_data)} questions")
    
    with open(f'{base_dir}/dev_answer.json', 'w') as f:
        json.dump(answers, f, indent=2)

if __name__ == "__main__":
    generate_answers()