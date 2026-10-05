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
    
    # Generation diagnostics counters
    diag = {
        "total": 0,
        "reached_a": 0,
        "reached_eos": 0,
        "truncated": 0,
        "parseable_answer": 0
    }

    if num_samples and num_samples < len(dataset):
        df_indices = pd.DataFrame({"source": dataset["source"], "idx": range(len(dataset))})
        sampled_indices = []
        sources = df_indices["source"].unique()
        for src in sources:
            src_df = df_indices[df_indices["source"] == src]
            n_src = max(1, int(round(num_samples * len(src_df) / len(dataset))))
            sampled_src = src_df.sample(n=min(n_src, len(src_df)), random_state=42)
            sampled_indices.extend(sampled_src["idx"].tolist())
        
        if len(sampled_indices) > num_samples:
            sampled_indices = sampled_indices[:num_samples]
        elif len(sampled_indices) < num_samples:
            remaining = list(set(range(len(dataset))) - set(sampled_indices))
            import random
            random.seed(42)
            sampled_indices.extend(random.sample(remaining, num_samples - len(sampled_indices)))
        
        sampled_indices.sort()
        dataset = dataset.select(sampled_indices)

    for item in tqdm(dataset, desc="Evaluating MathSLM Benchmark"):
        q = item["question"]
        gt_answer = item["answer"]
        subject = item.get("subject", "general")
        source = item.get("source", "unknown")

        if subject not in category_stats:
            category_stats[subject] = {"total": 0, "correct": 0}
        category_stats[subject]["total"] += 1

        prompt = f"[Q] {q} [R]"
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

        out_ids = generate(model, input_ids, max_new_tokens=256, temperature=0.0, device=device, eos_id=tokenizer.eos_id)
        out_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=False)

        # Update diagnostics
        diag["total"] += 1
        if "[A]" in out_text:
            diag["reached_a"] += 1
        if tokenizer.eos_id in out_ids[0].tolist():
            diag["reached_eos"] += 1
        else:
            diag["truncated"] += 1

        reasoning, pred_answer = parse_generated_text(out_text)
        if pred_answer:
            diag["parseable_answer"] += 1

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
            "source": source,
            "is_correct": is_correct
        })

    total_acc = correct_count / max(1, len(dataset))
    return total_acc, category_stats, results, diag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to checkpoint file")
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

    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif os.path.exists(best_ckpt):
        ckpt_path = best_ckpt
    elif os.path.exists(final_ckpt):
        ckpt_path = final_ckpt
    else:
        ckpt_path = None

    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        step = checkpoint.get("step", "N/A") if isinstance(checkpoint, dict) else "N/A"
        epoch = checkpoint.get("epoch", "N/A") if isinstance(checkpoint, dict) else "N/A"
        print(f"Loaded checkpoint from: {ckpt_path}")
        print(f"Checkpoint Metadata -> Step: {step}, Epoch: {epoch}")
        print(f"Model Parameters: {model.get_num_params():,} | Layers: {model_config.n_layer} | Heads: {model_config.n_head} | Embd: {model_config.n_embd}")
    else:
        print("Warning: No trained checkpoint found! Benchmarking UNTRAINED model baseline.")

    model.to(device)

    data_dir = config["paths"]["data_processed"]
    dataset_dict = load_from_disk(data_dir)
    test_ds = dataset_dict["test"]

    print(f"Running benchmark on held-out test split ({len(test_ds)} total samples available)...")
    accuracy, cat_stats, results, diag = run_benchmark(model, tokenizer, test_ds, device, num_samples=args.num_samples)

    print("\n" + "=" * 50)
    print(f"MATHSLM BENCHMARK RESULTS (Accuracy: {accuracy * 100:.2f}%)")
    print("=" * 50)
    for subj, stats in cat_stats.items():
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
        print(f" - {subj}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
    print("=" * 50)
    print("GENERATION DIAGNOSTICS:")
    print(f" - Total Samples Evaluated: {diag['total']}")
    print(f" - Reached [A] Tag: {diag['reached_a']}/{diag['total']} ({diag['reached_a']/max(1,diag['total'])*100:.1f}%)")
    print(f" - Reached [EOS] Tag: {diag['reached_eos']}/{diag['total']} ({diag['reached_eos']/max(1,diag['total'])*100:.1f}%)")
    print(f" - Truncated (Max Tokens): {diag['truncated']}/{diag['total']} ({diag['truncated']/max(1,diag['total'])*100:.1f}%)")
    print(f" - Non-empty Parseable Answer: {diag['parseable_answer']}/{diag['total']} ({diag['parseable_answer']/max(1,diag['total'])*100:.1f}%)")
    print("=" * 50)

    log_dir = config["paths"]["logs"]
    os.makedirs(log_dir, exist_ok=True)
    res_df = pd.DataFrame(results)
    out_csv = os.path.join(log_dir, "benchmark_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"Detailed benchmark predictions saved to {out_csv}")

if __name__ == "__main__":
    main()
