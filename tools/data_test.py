import json
import time
from typing import Callable, Dict, Any, List
from functools import partial
from transformers import AutoTokenizer
from openai import OpenAI, APIConnectionError, APIError
from tools.text import extract_box_content
import math
from tqdm import tqdm  # Import tqdm for progress bar
import os
import shutil
def evaluate_threshold(
    config: Dict[str, Any],
    data_file: str,
    threshold_fn: Callable[[str], int],
) -> None:
    """
    For each record in data_file (JSON or JSONL), run the judge_model on
    the full reasoning and on the first k tokens (where k = min(token_count, m)),
    then add a new field named threshold_fn.__name__ with {"k": k, "is_same": bool}.
    Writes out to <data_file>_evaluated.jsonl.
    """

    # --- 1. Load data ---
    with open(data_file, 'r', encoding='utf-8') as f:
        first = f.read(1)
        f.seek(0)
        if first == '[':
            # JSON array
            data: List[Dict] = json.load(f)
            is_jsonl = False
        else:
            # assume JSONL
            data = [json.loads(line) for line in f if line.strip()]
            is_jsonl = True
    backup_path = data_file.rsplit('.', 1)[0] + "_backup.jsonl"
    if os.path.exists(data_file):
        shutil.copyfile(data_file, backup_path)
        print(f"Backup saved to {backup_path}")
    # --- 2. Set up tokenizer and clients ---
    model_for_tokenizer = config.get('original_model', config['judge_model'])
    tokenizer = AutoTokenizer.from_pretrained(model_for_tokenizer)

    client = OpenAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    if config.get('LLM_check', False):
        check_client = OpenAI(
            api_key=config['check_api'],
            base_url=config['check_url']
        )
    else:
        check_client = None

    def normalize_answer(s: str) -> str:
        return s.strip().lower().replace(' ', '').replace('$', '')

    def llm_validate(ans1: str, ans2: str) -> bool:
        prompt = (
            f"Are these two mathematical answers equivalent?\n"
            f"Answer 1: {ans1}\n"
            f"Answer 2: {ans2}\n"
            "Respond with only \"yes\" or \"no\"."
        )
        resp = check_client.chat.completions.create(
            model=config['judge_model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=35.0
        )
        return "yes" in resp.choices[0].message.content.lower()

    def get_answer(problem: str, reasoning: str) -> str:
        prompt = (
            "Here is a problem and a portion of the reasoning process.\n"
            "Please directly derive the final answer based ONLY on the given problem and reasoning segment.\n"
            "Return your final response within \\boxed{} or \\boxed{None} if no conclusion can be drawn.\n"
            "-----------------------------------\n"
            f"Problem: {problem}\n"
            f"Reasoning segment: {reasoning}\n"
        )
        # retry logic
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=config['judge_model'],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user",   "content": prompt}
                    ],
                    temperature=0.0,
                    timeout=60.0
                )
                return resp.choices[0].message.content
            except APIConnectionError:
                time.sleep(2 ** attempt)
                continue
            except APIError as e:
                if e.status_code == 429 and attempt < 2:
                    time.sleep(5 ** attempt)
                    continue
                break
        return ""

    # --- 3. Process each entry with progress bar ---
    key_name = threshold_fn.__name__
    
    with open(data_file, 'w', encoding='utf-8') as f_out:
        key_name = threshold_fn.__name__

        with tqdm(data, desc="Processing entries", unit="entry") as progress_bar:
            for entry in progress_bar:
                prob = entry.get('problem')
                full = entry.get('full_reasoning', "")
                m = threshold_fn(entry=entry)

                if m:
                    token_ids = tokenizer.encode(full, add_special_tokens=False)
                    k = math.ceil(min(len(token_ids), m))
                    partial = tokenizer.decode(token_ids[:k], skip_special_tokens=True)

                    full_ans = get_answer(prob, full)
                    partial_ans = get_answer(prob, partial)

                    f_lst = extract_box_content(full_ans)
                    p_lst = extract_box_content(partial_ans)
                    is_same = False
                    if f_lst and p_lst and "None" not in (p_lst[-1] or f_lst[-1]):
                        norm_f = normalize_answer(f_lst[-1])
                        norm_p = normalize_answer(p_lst[-1])
                        if norm_f == norm_p or norm_f in norm_p or norm_p in norm_f:
                            is_same = True
                        elif check_client:
                            is_same = llm_validate(norm_p, norm_f)

                    entry[key_name] = {
                        "k": k,
                        "is_same": is_same
                    }
                    entry["original_length"] = len(token_ids)
                else:
                    entry[key_name] = "invalid threshold"

                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Done. Results written to {data_file}")


def NC_0(tao=None,entry=None):
    return tao

def NC_1(tao_list = None,entry = None):
    """
    conditional conformal prediction, for different difficulty level, we use the diffrent threshold
    """
    level = entry.get('pred_level',None)
    if level:
        return tao_list[int(level)-1]
    return None

def NC_2(tao=None,entry=None):
    """
    the nonconformity score is l_min/pred_length
    """
    pred_length = entry.get("pred_length",None)
    return math.ceil(tao*pred_length) 