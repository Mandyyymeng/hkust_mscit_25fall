#! /bin/bash

DB_ROOT_DIR="data/bird/dev/dev_databases"
PROCESS_NUM=16

RESULTS_DIR="results/Qwen2.5-Coder-32B-Instruct/bird/dev"  #Qwen2.5-Coder-7B-Instruct
OUTPUT_PATH="./pred_sqls_qwen32b_bird_300.json"

echo "Selecting SQLs..."
python -m alphasql.runner.sql_selection \
    --results_dir $RESULTS_DIR \
    --db_root_dir $DB_ROOT_DIR \
    --process_num $PROCESS_NUM \
    --output_path $OUTPUT_PATH 
