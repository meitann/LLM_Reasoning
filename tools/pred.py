import json
import re
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from tools.Prompt import PREDICT_DIFFICULT_LEVEL,PREDICT_REASONING_LENGTH

def extract_level(text: str):
    match = re.search(r"\\boxed\{(\d)\}", text)
    return int(match.group(1)) if match else None

def extract_number(text: str):
    match = re.search(r"\\boxed\{(\d+)\}", text)
    return int(match.group(1)) if match else None

def query_with_retry(client, prompt,model_name = "deepseek-chat", retries=3, delay=10):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Retry {attempt + 1}/{retries}] Error: {e}")
            time.sleep(delay)
    return None

def annotate_difficulty_level(input_path: str, api_key: str, base_url: str,model_name:str, retries: int = 3, delay: int = 5):
    input_path = Path(input_path)
    assert input_path.exists(), f"File not found: {input_path}"

    output_path = input_path.with_name(input_path.stem + "-level.jsonl")

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}")

    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in tqdm(infile, desc="Annotating difficulty level"):
            data = json.loads(line)
            problem = data.get("problem", "")

            prompt = PREDICT_DIFFICULT_LEVEL.format(problem=problem)
            reply = query_with_retry(client, prompt, model_name=model_name,retries=retries, delay=delay)

            data["pred_level"] = extract_level(reply) if reply else None
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
    backup_path = input_path.with_suffix(".jsonl.bak")
    input_path.replace(backup_path)
    output_path.replace(input_path)
    print("✅ Annotation complete.")


def predict_reasoning_length(input_path: str, api_key: str, base_url: str,model_name:str, retries: int = 3, delay: int = 5):
    input_path = Path(input_path)
    assert input_path.exists(), f"File not found: {input_path}"

    output_path = input_path.with_name(input_path.stem + "-level.jsonl")

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}")

    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in tqdm(infile, desc="Predictiong reasoing length"):
            data = json.loads(line)
            problem = data.get("problem", "")

            prompt = PREDICT_REASONING_LENGTH.format(new_problem=problem)
            reply = query_with_retry(client, prompt, model_name=model_name,retries=retries, delay=delay)

            data["pred_length"] = extract_number(reply) if reply else None
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
    backup_path = input_path.with_suffix(".jsonl.bak")
    input_path.replace(backup_path)
    output_path.replace(input_path)
    print("✅ prediction complete.")
