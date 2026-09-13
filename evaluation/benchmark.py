import argparse
import os
import sys
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_from_disk

from model.config import MathSLMConfig
from model.transformer import MathSLM
from model.generation import generate, parse_generated_text
from tokenizer.tokenizer import MathTokenizer
from evaluation.metrics import calculate_exact_match
from training.train import get_device

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_benchmark(model, tokenizer, dataset, device, num_samples=100):
    model.eval()
    results = []
    correct_count = 0
    category_stats = {}

    dataset = dataset.select(range(min(num_samples, len(dataset))))

    for item in tqdm(dataset, desc="Evaluating MathSLM Benchmark"):
        q = item["question"]
        gt_answer = item["answer"]
        subject = item.get("subject", "general")

        if subject not in category_stats:
            category_stats[subject] = {"total": 0, "correct": 0}
        category_stats[subject]["total"] += 1

        prompt = f"[Q] {q} [R]"
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

        out_ids = generate(model, input_ids, max_new_tokens=256, temperature=0.0, device=device, eos_id=tokenizer.eos_id)
        out_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=False)

        reasoning, pred_answer = parse_generated_text(out_text)
        is_correct = calculate_exact_match(pred_answer, gt_answer)

        if is_correct:
            correct_count += 1
            category_stats[subject]["correct"] += 1

        results.append({
            "question": q,
            "ground_truth": gt_answer,
            "predicted_answer": pred_answer,
            "reasoning": reasoning,
            "subject": subject,
            "is_correct": is_correct
        })

    total_acc = correct_count / max(1, len(dataset))
    return total_acc, category_stats, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cpu" if config.get("smoke_test", False) else get_device(config["training"].get("device", "auto"))

    tokenizer_path = config["paths"]["tokenizer"]
    tokenizer = MathTokenizer.load(tokenizer_path)

    config["model"]["vocab_size"] = tokenizer.vocab_size
    model_config = MathSLMConfig.from_dict(config["model"])
    model = MathSLM(model_config)

    ckpt_dir = config["paths"]["checkpoints"]
    best_ckpt = os.path.join(ckpt_dir, "model_best.pt")
    final_ckpt = os.path.join(ckpt_dir, "model_final.pt")

    if os.path.exists(best_ckpt):
        ckpt_path = best_ckpt
    elif os.path.exists(final_ckpt):
        ckpt_path = final_ckpt
    else:
        ckpt_path = None

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("Warning: No trained checkpoint found! Benchmarking UNTRAINED model baseline.")

    model.to(device)

    data_dir = config["paths"]["data_processed"]
    dataset_dict = load_from_disk(data_dir)
    test_ds = dataset_dict["test"]

    print(f"Running benchmark on held-out test split ({len(test_ds)} total samples available)...")
    accuracy, cat_stats, results = run_benchmark(model, tokenizer, test_ds, device, num_samples=args.num_samples)

    print("\n" + "=" * 50)
    print(f"MATHSLM BENCHMARK RESULTS (Accuracy: {accuracy * 100:.2f}%)")
    print("=" * 50)
    for subj, stats in cat_stats.items():
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
        print(f" - {subj}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
    print("=" * 50)

    log_dir = config["paths"]["logs"]
    os.makedirs(log_dir, exist_ok=True)
    res_df = pd.DataFrame(results)
    out_csv = os.path.join(log_dir, "benchmark_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"Detailed benchmark predictions saved to {out_csv}")

if __name__ == "__main__":
    main()
