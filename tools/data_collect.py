# from openai import OpenAI
from tools import Prompt
import os
from pathlib import Path
# import openai

# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com/v1" 
# )

# response = client.chat.completions.create(
#     model="deepseek-reasoner",
#     messages=[
#         {"role": "system", "content": Prompt.DEEPSEEK_R1_SYSTEM_PROMPT},
#         {"role": "user", "content": }
#     ],
#     temperature=0.0,  
#     max_tokens=5000   
# )
import os
import json
import queue
from typing import Dict,List
import threading
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
import time
import os 
class Data_Processor:
    def __init__(
        self,
        api_config: Dict,
        retries: int = 2,
        max_workers: int = 10,
        timeout: int = 30,
    ):
        # init the client
        self.api_cofig = api_config
        self.api_key = api_config['api_key']
        self.client = OpenAI(
            api_key=api_config['api_key'],
            base_url=api_config['base_url']
        )
        
        self.max_workers = max_workers
        self.output_file = setup_output_file(api_config)
        self.retries = retries
        self.timeout = timeout
        self.result_lock = threading.Lock()
        
        self.progress = 0
        self.progress_bar = None

    def _build_messages(self, problem: str) -> list:
        return [
            {"role": "system", "content": Prompt.DEEPSEEK_R1_SYSTEM_PROMPT},
            {"role": "user", "content": f"Return your final response within \\boxed{{}}. {problem}"}
            # {"role": "user", "content": problem}
        ]

    def _process_single(self, problem: str) -> dict:
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                    model = self.api_cofig['model'],
                    messages=self._build_messages(problem),
                    temperature=0.0,
                    max_tokens=8000,
                    timeout=300
                    # top_p=1.0
                )
                
                return {
                    "problem": problem,
                    "full_reasoning": response.choices[0].message.reasoning_content,
                    "result": response.choices[0].message.content,
                    "status": "success"
                }
            except Exception as e:
                if attempt == self.retries - 1:
                    return {
                        "problem": problem,
                        "error": str(e),
                        "status": "failed"
                    }
                time.sleep(2 ** attempt)

    def _worker(self, task_queue: queue.Queue):
        while True:
            problem = task_queue.get()
            if problem is None:
                break
                
            result = self._process_single(problem)
            
            # with self.result_lock, open(self.output_file, "a") as f:
            #     f.write(json.dumps(result, ensure_ascii=False) + "\n")
            with self.result_lock:
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            with self.result_lock:
                self.progress += 1
                if self.progress_bar:
                    self.progress_bar.update(1)
                    
            task_queue.task_done()

    def process_dataset(self, dataset, batch_size: int = 1000):
        task_queue = queue.Queue(maxsize=batch_size)

        threads = []
        for _ in range(self.max_workers):
            t = threading.Thread(target=self._worker, args=(task_queue,))
            t.start()
            threads.append(t)
        
        total = min(len(dataset), batch_size)
        self.progress_bar = tqdm(total=total, desc="Processing")
        
        for i, example in enumerate(dataset):
            if i >= batch_size:
                break
            if example.get("problem",None):
                task_queue.put(example["problem"])
            else:
                task_queue.put(example["Problem"])
        task_queue.join()
        
        for _ in range(self.max_workers):
            task_queue.put(None)
        for t in threads:
            t.join()
        
        self.progress_bar.close()




def setup_output_file(api_config: dict) -> Path:
    output_file = api_config.get('output_file', f"reasoning_datasets/{api_config['dataset_name']}.jsonl")
    output_path = Path(output_file)
    
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return  output_path