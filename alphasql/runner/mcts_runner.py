from alphasql.algorithm.mcts.mcts import MCTSSolver
from alphasql.algorithm.mcts.reward import MajorityVoteRewardModel
from alphasql.runner.task import Task
from alphasql.config.mcts_config import MCTSConfig
from pathlib import Path
from typing import Union
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import yaml
from alphasql.llm_call.openai_llm import DEFAULT_COST_RECORDER
import json
import random
from dotenv import load_dotenv  
import os
import traceback
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

load_dotenv(override=True)

try:
    import weave
    if "WANDB_API_KEY" in os.environ:
        weave.init("mcts_runner")
        logger.info("Weave initialized successfully")
    else:
        logger.warning("WANDB_API_KEY is not set in environment variables, will not use it")
except ImportError:
    logger.warning("Weave is not installed, will not use it")

class MCTSRunner:
    def __init__(self, config: Union[MCTSConfig, str]):
        self.start_time = time.time()
        
        if isinstance(config, str):
            config_path = Path(config)
            assert config_path.exists(), f"Config file {config_path} does not exist"
            if config_path.suffix == ".json":
                self.config = MCTSConfig.model_validate_json(config_path.read_text())
            elif config_path.suffix == ".yaml":
                self.config = MCTSConfig.model_validate(yaml.safe_load(config_path.read_text()))
            else:
                raise ValueError(f"Unsupported config file extension: {config_path.suffix}")
        else:
            self.config = config
            
        if not Path(self.config.save_root_dir).exists():
            Path(self.config.save_root_dir).mkdir(parents=True, exist_ok=True)

        random.seed(self.config.random_seed)
        
        logger.info(f"MCTSRunner initialized with config: {self.config.model_dump()}")
        
    def run_one_task(self, task: Task) -> str:
        task_start_time = time.time()
        logger.info(f"Starting task {task.question_id}: {task.question[:100]}...")
        
        mcts_solver = MCTSSolver(
            db_root_dir=self.config.db_root_dir,
            task=task,
            max_rollout_steps=self.config.max_rollout_steps,
            max_depth=self.config.max_depth,
            exploration_constant=self.config.exploration_constant,
            save_root_dir=self.config.save_root_dir,
            llm_kwargs=self.config.mcts_model_kwargs,
            reward_model=MajorityVoteRewardModel(self.config.reward_model_kwargs)
        )
        try:
            mcts_solver.solve()
            task_duration = time.time() - task_start_time
            logger.info(f"Task {task.question_id} completed successfully in {task_duration:.2f}s")
        except Exception as e:
            task_duration = time.time() - task_start_time
            logger.error(f"Error solving task {task.question_id} after {task_duration:.2f}s: {str(e)}")
            logger.debug(f"Traceback for task {task.question_id}: {traceback.format_exc()}")
        
        DEFAULT_COST_RECORDER.print_profile()
    
    def run_all_tasks(self):
        total_start_time = time.time()
        logger.info("Starting MCTS run for all tasks")
        
        with open(self.config.tasks_file_path, "rb") as f:
            tasks = pickle.load(f)
            
        if self.config.subset_file_path:
            logger.info(f"Using subset file {self.config.subset_file_path} to filter tasks")
            with open(self.config.subset_file_path, "r") as f:
                subset_data = json.load(f)
                subset_ids = [item["question_id"] for item in subset_data]
                tasks = [task for task in tasks if task.question_id in subset_ids]
            logger.info(f"Filtered {len(tasks)} tasks from original set")

        done_task_ids = []
        for pkl_file in Path(self.config.save_root_dir).glob("*.pkl"):
            done_task_ids.append(int(pkl_file.stem))
        logger.info(f"Found {len(done_task_ids)} completed tasks, will skip them")
        
        tasks = [task for task in tasks if task.question_id not in done_task_ids]
        
        with open(Path(self.config.save_root_dir) / "config.json", "w") as f:
            logger.info(f"Saving config to {Path(self.config.save_root_dir) / 'config.json'}")
            json.dump(self.config.model_dump(), f, indent=4)

        logger.info(f"Total tasks to solve: {len(tasks)}")
        
        if len(tasks) == 0:
            logger.info("No tasks to solve, exiting")
            return
            
        process_start_time = time.time()
        with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
            list(tqdm(executor.map(self.run_one_task, tasks), total=len(tasks), desc="Solving tasks"))
        
        process_duration = time.time() - process_start_time
        total_duration = time.time() - total_start_time
        
        logger.info(f"All tasks completed. Process time: {process_duration:.2f}s, Total time: {total_duration:.2f}s")
        
        # Log summary statistics
        completed_files = list(Path(self.config.save_root_dir).glob("*.pkl"))
        logger.info(f"Final completion status: {len(completed_files)} tasks completed out of {len(tasks) + len(done_task_ids)} total")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        logger.error("Usage: python -m alphasql.runner.mcts_runner <config_path>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    logger.info(f"Starting MCTS Runner with config: {config_path}")
    
    try:
        runner = MCTSRunner(config=config_path)
        runner.run_all_tasks()
        logger.info("MCTS Runner completed successfully")
    except Exception as e:
        logger.error(f"MCTS Runner failed: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)