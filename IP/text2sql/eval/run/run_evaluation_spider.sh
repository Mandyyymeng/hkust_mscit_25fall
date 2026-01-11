db_root_path='../Alpha-SQL/data/spider/test_database/'
data_mode='dev'
diff_json_path='../Alpha-SQL/data/spider/dev.json'  # get difficulty for json key "difficulty" 'data/new_dev.json'
# predicted_sql_path_kg='./exp_result/turbo_output_kg/'
predicted_sql_path='./exp_result/spider_test_alpha_qwen32b/'    #rslsql_output_new/' # rslsql_output # bird_dev_300_new
ground_truth_path='../Alpha-SQL/data/spider/' # ./data/
num_cpus=16
meta_time_out=30.0 
mode_gt='gt'  # use dev_gold.sql
mode_predict='gpt'  # if "gpt", use predict_dev.json as ground truth

echo '''starting to compare for ex'''
python3 -u ./src/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

echo '''starting to compare for ves'''
python3 -u ./src/evaluation_ves_cp.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}
 
 