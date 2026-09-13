import argparse
import io
import json
import os
import re
import urllib.request
import zipfile
import yaml
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Normalize excessive spaces but keep standard math spaces
    text = re.sub(r'[ \t]+', ' ', text).strip()
    return text

def extract_answer_gsm8k(answer_str):
    parts = answer_str.split("####")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return answer_str.strip(), ""

def compute_difficulty(question, reasoning, source):
    score = 0
    if len(question) > 200: score += 1
    if len(reasoning) > 400: score += 2
    if source in ["gsm8k", "metamath"]: score += 1
    elif source == "mathqa": score += 2
    elif source == "math": score += 3
    else: score += 1
    
    if score <= 2: return "easy"
    elif score <= 4: return "medium"
    else: return "hard"

def process_gsm8k(smoke_test=False):
    print("Processing GSM8K...")
    try:
        dataset = load_dataset("openai/gsm8k", "main")
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        return []
    
    if smoke_test:
        dataset["train"] = dataset["train"].select(range(min(100, len(dataset["train"]))))
        dataset["test"] = dataset["test"].select(range(min(20, len(dataset["test"]))))
        
    processed_data = []
    for split in ["train", "test"]:
        for item in tqdm(dataset[split], desc=f"GSM8K {split}"):
            q = clean_text(item["question"])
            reasoning, final_answer = extract_answer_gsm8k(item["answer"])
            reasoning = clean_text(reasoning)
            final_answer = clean_text(final_answer)
            diff = compute_difficulty(q, reasoning, "gsm8k")
            
            processed_data.append({
                "question": q,
                "reasoning": reasoning,
                "answer": final_answer,
                "difficulty": diff,
                "source": "gsm8k",
                "subject": "word_problems",
                "split": split
            })
            
    return processed_data

def process_math(smoke_test=False):
    print("Processing Hendrycks MATH dataset...")
    categories = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    processed_data = []
    
    for cat in categories:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", cat)
            for split in ["train", "test"]:
                items = ds[split]
                if smoke_test:
                    items = items.select(range(min(20, len(items))))
                for item in items:
                    q = clean_text(item["problem"])
                    sol = item["solution"]
                    boxed_matches = re.findall(r'\\boxed{(.*?)}', sol)
                    final_answer = boxed_matches[-1] if boxed_matches else ""
                    reasoning = clean_text(sol)
                    final_answer = clean_text(final_answer)
                    diff = compute_difficulty(q, reasoning, "math")
                    
                    processed_data.append({
                        "question": q,
                        "reasoning": reasoning,
                        "answer": final_answer,
                        "difficulty": diff,
                        "source": "math",
                        "subject": cat,
                        "split": split
                    })
        except Exception as e:
            print(f"Warning: Could not process MATH category {cat}: {e}")
            
    return processed_data

def process_mathqa(smoke_test=False):
    print("Processing MathQA...")
    url = "https://math-qa.github.io/math-QA/data/MathQA.zip"
    processed_data = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            zip_bytes = response.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                splits = [("train.json", "train"), ("test.json", "test"), ("dev.json", "train")]
                for fname, split_name in splits:
                    if fname in z.namelist():
                        content = z.read(fname).decode('utf-8', errors='replace')
                        raw_items = json.loads(content)
                        if smoke_test:
                            raw_items = raw_items[:100 if split_name == "train" else 20]
                        for item in tqdm(raw_items, desc=f"MathQA {fname}"):
                            q = clean_text(item.get("Problem", ""))
                            reasoning = clean_text(item.get("Rationale", ""))
                            final_answer = clean_text(str(item.get("correct", "")))
                            cat = item.get("category", "general")
                            diff = compute_difficulty(q, reasoning, "mathqa")
                            
                            processed_data.append({
                                "question": q,
                                "reasoning": reasoning,
                                "answer": final_answer,
                                "difficulty": diff,
                                "source": "mathqa",
                                "subject": cat,
                                "split": split_name
                            })
    except Exception as e:
        print(f"Warning: Could not download/process MathQA: {e}")
        
    return processed_data

def process_metamath(smoke_test=False, max_samples=20000):
    print("Processing MetaMathQA...")
    processed_data = []
    try:
        ds = load_dataset("meta-math/MetaMathQA", split="train")
        if smoke_test:
            ds = ds.select(range(min(200, len(ds))))
        elif max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
            
        for item in tqdm(ds, desc="MetaMathQA"):
            q = clean_text(item.get("query", ""))
            sol = item.get("response", "")
            boxed_matches = re.findall(r'\\boxed{(.*?)}', sol)
            if not boxed_matches:
                boxed_matches = re.findall(r'The answer is:?\s*([^\n\.]+)', sol, re.IGNORECASE)
            final_answer = boxed_matches[-1] if boxed_matches else ""
            reasoning = clean_text(sol)
            final_answer = clean_text(final_answer)
            diff = compute_difficulty(q, reasoning, "metamath")
            
            processed_data.append({
                "question": q,
                "reasoning": reasoning,
                "answer": final_answer,
                "difficulty": diff,
                "source": "metamath",
                "subject": "reasoning",
                "split": "train"
            })
    except Exception as e:
        print(f"Warning: Could not process MetaMathQA: {e}")
        
    return processed_data

def normalize_text_for_dedup(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

def deduplicate_data(data_list):
    print("Performing exact and normalized-text deduplication...")
    test_questions = set()
    for item in data_list:
        if item["split"] == "test":
            norm_q = normalize_text_for_dedup(item["question"])
            if norm_q:
                test_questions.add(norm_q)
                
    deduped = []
    seen_train_questions = set()
    removed_leakage_count = 0
    removed_duplicate_count = 0
    
    for item in data_list:
        norm_q = normalize_text_for_dedup(item["question"])
        if not norm_q or not item["answer"]:
            continue
            
        if item["split"] == "test":
            deduped.append(item)
        else:
            if norm_q in test_questions:
                removed_leakage_count += 1
                continue
            if norm_q in seen_train_questions:
                removed_duplicate_count += 1
                continue
            seen_train_questions.add(norm_q)
            deduped.append(item)
            
    print(f"Deduplication complete. Removed {removed_duplicate_count} duplicates and {removed_leakage_count} potential test-set leaks.")
    return deduped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    smoke_test = config.get("smoke_test", False)
    
    out_dir = config["paths"]["data_processed"]
    os.makedirs(out_dir, exist_ok=True)
    
    all_data = []
    all_data.extend(process_gsm8k(smoke_test))
    all_data.extend(process_math(smoke_test))
    all_data.extend(process_mathqa(smoke_test))
    all_data.extend(process_metamath(smoke_test, max_samples=500 if smoke_test else 25000))
    
    all_data = deduplicate_data(all_data)
    
    df = pd.DataFrame(all_data)
    
    test_df = df[df["split"] == "test"].copy()
    train_val_df = df[df["split"] == "train"].copy()
    
    val_frac = 0.1
    val_df = train_val_df.sample(frac=val_frac, random_state=42)
    train_df = train_val_df.drop(val_df.index)
    
    print("Saving processed datasets...")
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    })
    
    dataset_dict.save_to_disk(out_dir)
    print(f"Datasets successfully saved to {out_dir}")
    print(f"Summary: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

if __name__ == "__main__":
    main()
