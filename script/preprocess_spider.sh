#! /bin/bash
# automatically saved to save_root_dir/dev (data_split)
python -m alphasql.runner.preprocessor \
    --data_file_path "data/spider/dev.json" \
    --database_root_dir "data/spider/test_database" \
    --save_root_dir "data/preprocessed/spider/test/qwen32b" \
    --lsh_threshold 0.5 \
    --lsh_signature_size 128 \
    --lsh_n_gram 3 \
    --lsh_top_k 20 \
    --edit_similarity_threshold 0.3 \
    --embedding_similarity_threshold 0.6 \
    --n_parallel_processes 8 \
    --max_dataset_samples -1