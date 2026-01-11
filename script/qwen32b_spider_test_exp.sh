#! /bin/bash

start_time=$(date +%s)

echo "Running MCTS..."
python -m alphasql.runner.mcts_runner config/qwen32b_spider_test.yaml  # dschat_arch.yaml

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"
