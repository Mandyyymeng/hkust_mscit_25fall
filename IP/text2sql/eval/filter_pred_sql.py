import json

def filter_and_sort_json_by_question_id(dev_json_path, predict_dev_json_path):
    """
    Filter and sort predict_dev.json based on question_id in dev.json
    
    Args:
        dev_json_path: Path to dev.json containing question_id
        predict_dev_json_path: Path to predict_dev.json to be filtered and sorted
    """
    # Read dev.json to get question_id list
    with open(dev_json_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    # Extract question_id values and convert to string (as JSON keys are strings)
    question_ids = set()
    for item in dev_data:
        if 'question_id' in item:
            question_ids.add(str(item['question_id']))
    
    print(f"Found {len(question_ids)} unique question_ids in dev.json")
    
    # Read predict_dev.json
    with open(predict_dev_json_path, 'r', encoding='utf-8') as f:
        predict_data = json.load(f)
    
    # Filter predict_dev.json - only keep items with keys that exist in dev.json question_ids
    filtered_data = {}
    for key, value in predict_data.items():
        if key in question_ids:
            filtered_data[key] = value
    
    print(f"Filtered from {len(predict_data)} to {len(filtered_data)} items")
    
    # Sort by integer value of keys
    sorted_keys = sorted(filtered_data.keys(), key=lambda x: int(x))
    sorted_data = {key: filtered_data[key] for key in sorted_keys}
    
    # Write back to the original predict_dev.json file
    with open(predict_dev_json_path.split('.')[0].strip()+"_0.json", 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully filtered, sorted and saved {len(sorted_data)} items to {predict_dev_json_path}")

if __name__ == "__main__":
    # File paths
    dev_json_path = "../Alpha-SQL/data/bird/dev/dev.json"
    predict_dev_json_path = "exp_result/bird_dev_300/predict_dev.json"
    
    # Execute the filtering and sorting
    filter_and_sort_json_by_question_id(dev_json_path, predict_dev_json_path)