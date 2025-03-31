import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import concurrent
import re
from typing import Dict, List, Optional
from openai import OpenAI, APIConnectionError, APIError
from tools import Prompt
from tqdm import tqdm
import time
from pathlib import Path
from threading import Lock
from tools.text import extract_box_content,extract_box_number
import traceback
class DataEntry:
    def __init__(self, entry_id: str, problem: str, full_reasoning: str, result: str):
        self.id = entry_id
        self.problem = problem
        self.full_reasoning = full_reasoning  
        self.result = result  
        self.pred_length = None
        self.min_valid_length: Optional[int] = None  
        self.min_valid_percent: Optional[int] = None  



class ReasoningOptimizer:
    def __init__(self, api_config: Dict,output_path = None):
        self.api_config = api_config
        self.entries: List[DataEntry] = []
        if(output_path):
            self.output_path = Path(output_path)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("Please give an output file")
        self.client = OpenAI(
            api_key=api_config['api_key'],
            base_url = api_config['base_url']
        )
        self.pred_client = OpenAI(
            api_key=api_config['api_key'],
            base_url = api_config['base_url']
        )
        if api_config['LLM_check']:
            self.check_client = OpenAI(
                api_key=api_config['check_api'],
                base_url = api_config['check_url']
            )
        self.write_lock = Lock() 




    def load_data(self, jsonl_path: str):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                if(data['status']=='failed'):
                    continue
                self.entries.append(
                    DataEntry(
                        entry_id=f"entry_{idx}",
                        problem=data['problem'],
                        full_reasoning=data['full_reasoning'],
                        result=data['result']
                    )
                )

    def query_pred_length(self,problem:str) -> str:
        max_retries = 2
        prompt =  f"""Here is a problem
You need to make a brief assessment of the difficulty of the problem, then estimate how many tokens a reasonably efficient reasoning model would need to complete the reasoning process.
Finally, output an integer within \\boxed{{}}, representing the expected reasoning length for the reasoning model.
-----------------------------------
Here is the target problem:
Problem: {problem}
"""

        for _ in range(max_retries):
            try:
                response = self.pred_client.chat.completions.create(
                    model=self.api_config['pred_model'] ,
                    messages=[{"role": "system", "content":"You are helpful assistant, now you need to complete user's task."},{"role": "user", "content": prompt}],
                    temperature=0.0,
                    timeout=35.0
                )
                content = response.choices[0].message.content
                return extract_box_number(content)
            
            except APIConnectionError as e:
                print('APIConnectionError')
                continue
            except APIError as e:
                if e.status_code == 429:  # speed constraint
                    time.sleep(1) 
                    print("APIError")
                    continue
                break 
            except (KeyError, AttributeError):
                break 
        return ""
    def query_api(self, problem: str, partial_reasoning: str) -> str:
        max_retries = 3
        prompt =  f"""Here is a problem and a portion of the reasoning process.
Please directly derive the final answer based ONLY on the given problem and reasoning segment.
Respond according to these requirements:
1. If you can determine a clear answer from the reasoning segment, Return your final response within \\boxed{{}}
2. If no conclusion can be drawn from the reasoning segment, output \\boxed{{None}} exactly
-----------------------------------
Problem: {problem}
Reasoning segment: {partial_reasoning}
"""

        for _ in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.api_config['model'] ,
                    messages=[{"role": "system", "content":"You are helpful assistant, now you need to complete user's task."},{"role": "user", "content": prompt}],
                    temperature=0.0,
                    timeout=35.0
                )
                return response.choices[0].message.content
            except APIConnectionError as e:
                # print('APIConnectionError')
                continue
            except APIError as e:
                if e.status_code == 429:  # speed constraint
                    time.sleep(1) 
                    print("APIError")
                    continue
                break 
            except (KeyError, AttributeError):
                break 
        return ""

    def split_reasoning(self, full_reasoning: str) -> Dict[int, int]:
        total_len = len(full_reasoning)
        return {p: max(1, int(total_len * p / 100)) for p in range(1, 101)}

    def is_result_valid(self, generated: str, target: str) -> bool:

        gen_match = extract_box_content(generated)
        target_match = extract_box_content(target)
        if len(gen_match)==0 or len(target_match)==0:
            return False
        

        gen_val = self._normalize_answer(gen_match[-1])
        target_val = self._normalize_answer(target_match[-1])
        if gen_val == target_val or (gen_val in target_val) or (target_val in gen_val):
            return True
        if self.api_config['LLM_check']:
            return self._llm_validate(gen_val, target_val)
        return False
    
    def _normalize_answer(self, s: str) -> str:
        return s.strip().lower().replace(' ', '').replace('$', '')
    


    def _llm_validate(self, generated: str, target: str) -> bool:
        validation_prompt = f"""Are these two mathematical answers equivalent?
        Answer 1: {generated}
        Answer 2: {target}
        Respond with only "yes" or "no":"""
        response = self.check_client.completions.create(
                    model=self.api_config['model'] ,
                    messages=[
                        {"role": "system", "content": Prompt.DEEPSEEK_R1_SYSTEM_PROMPT},
                        {"role": "user", "content": validation_prompt}],
                    temperature=0.0,
                    timeout=10.0
                )
        return "yes" in response.lower()
    
    def process_single_entry(self, entry: DataEntry):
        chunks = self.split_reasoning(entry.full_reasoning)
        
        low, high = 1, 100
        best_percent = 100
        while low <= high:
            mid = (low + high) // 2
            generated = self.query_api(entry.problem, entry.full_reasoning[:chunks[mid]])
        
            if self.is_result_valid(generated, entry.result):
                best_percent = mid
                high = mid - 1  
            else:
                low = mid + 1 
        
        entry.min_valid_percent = best_percent
        entry.min_valid_length = chunks[best_percent]
        entry.pred_length = self.query_pred_length(entry.problem)
        self._save_single_entry(entry)



    def run_optimization(self):
        max_workers = min(10, len(self.entries)) 
        rate_limit_delay = 0.1
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = (
                executor.submit(self._process_with_retry, entry)
                for entry in self.entries
            )

            with tqdm(total=len(self.entries), desc="Processing") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
               
                    except Exception as e:
                        self._log_error(e)
                    finally:
                        pbar.update(1)
                        time.sleep(rate_limit_delay) 

    def _process_with_retry(self, entry: DataEntry):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self.process_single_entry(entry)
            except APIError as e:
                if e.status_code == 429 and attempt < max_retries - 1:
                    time.sleep(2 ** attempt) 
                    continue
                raise
                
    def _log_error(self, error: Exception):
        error_type = type(error).__name__
        print(f"\n[ERROR] {error_type}: {str(error)}")
        if isinstance(error, APIError):
            print(f"API Status: {error.status_code}")

    def save_results(self, output_path: str):
        output_data = []
        for entry in self.entries:
            entry_dict = {
                "entry_id": entry.id,
                "problem": entry.problem,
                "original_reasoning_length": len(entry.full_reasoning),
                "min_valid_percent": entry.min_valid_percent,
                "min_valid_length": entry.min_valid_length,
                "target_result": entry.result
            }
            output_data.append(entry_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def _save_single_entry(self, entry: DataEntry):
            entry_data = {
                "entry_id": entry.id,
                "problem": entry.problem,
                "pred_length": entry.pred_length,
                "original_reasoning_length": len(entry.full_reasoning),
                "min_valid_percent": entry.min_valid_percent,
                "min_valid_length": entry.min_valid_length,
                "target_result": entry.result
                
            }
            line = json.dumps(entry_data) + "\n"
            with self.write_lock:
                try:
                    with open(self.output_path, 'a', encoding='utf-8') as f:
                        f.write(line)
                        f.flush()
                except IOError as e:
                    self._log_error(e)
                    raise
