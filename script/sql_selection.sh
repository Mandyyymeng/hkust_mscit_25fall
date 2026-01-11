#! /bin/bash

DB_ROOT_DIR="data/spider/test_database"
PROCESS_NUM=16

RESULTS_DIR="results/Qwen2.5-Coder-7B-Instruct/spider/test"  #Qwen2.5-Coder-7B-Instruct
OUTPUT_PATH="logs/pred_sqls/pred_sqls_qwen7b_spidertest.json"

echo "Selecting SQLs..."
python -m alphasql.runner.sql_selection \
    --results_dir $RESULTS_DIR \
    --db_root_dir $DB_ROOT_DIR \
    --process_num $PROCESS_NUM \
    --output_path $OUTPUT_PATH 
