from datasets import load_dataset
from tools.data_collect import Data_Processor,setup_output_file
from tools.data_judge import ReasoningOptimizer
import json
import os
import numpy as np
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
########################################
#              collect data            #
########################################

config_path = os.path.join(script_dir, 'configs', 'collect_config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    collect_config = json.load(f)
ds = load_dataset(collect_config['dataset_name'], split="test")
ds_demo = ds.select(range(400, 500))
processor = Data_Processor(
    # api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_config= collect_config,
    max_workers=10,  
)
processor.output_file = "reasoning_datasets\HuggingFaceH4\MATH-500-test.jsonl"
# dataset_path = setup_output_file(collect_config)
processor.process_dataset(ds_demo, batch_size=1000)





###################################################
#         find minest reasoning length            #
###################################################
# dataset_path = setup_output_file(collect_config)
# judge_config_path = os.path.join(script_dir, 'configs', 'judge_config.json')

# with open(judge_config_path, 'r', encoding='utf-8') as f:
#     judge_config = json.load(f)
# judged_output = Path(script_dir) / 'judged_datasets' / f"{collect_config['dataset_name']}.jsonl"
# optimizer = ReasoningOptimizer(judge_config,output_path=judged_output)
# optimizer.load_data(dataset_path)
# optimizer.run_optimization()






