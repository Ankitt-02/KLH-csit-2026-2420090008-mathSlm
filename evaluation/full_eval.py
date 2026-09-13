import argparse
import math
import os
import sys
import json
import re
import yaml
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import MathSLMConfig
from model.transformer import MathSLM
from model.generation import generate, parse_generated_text, generate_self_consistency
from tokenizer.tokenizer import MathTokenizer
from evaluation.metrics import calculate_exact_match

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_qualitative_tests(model, tokenizer, device):
    test_cases = [
        {
            "category": "Simple Arithmetic",
            "problem": "What is 27 * 14?",
            "expected": "378"
        },
        {
            "category": "Multi-Step Arithmetic",
            "problem": "What is (15 + 25) * 8 - 40?",
            "expected": "280"
        },
        {
            "category": "Algebra",
            "problem": "Solve 2x + 7 = 19.",
            "expected": "6"
        },
        {
            "category": "Percentages",
            "problem": "What is 15% of 200?",
            "expected": "30"
        },
        {
            "category": "Ratios",
            "problem": "If the ratio of boys to girls is 3:2 and there are 15 boys, how many girls are there?",
            "expected": "10"
        },
        {
            "category": "Geometry",
            "problem": "What is the area of a rectangle with length 8 and width 5?",
            "expected": "40"
        },
        {
            "category": "Probability",
            "problem": "What is the probability of rolling a 4 on a fair six-sided die?",
            "expected": "1/6"
        },
        {
            "category": "Quadratic",
            "problem": "Solve x^2 - 5x + 6 = 0.",
            "expected": "2, 3"
        },
        {
            "category": "GSM8K Word Problem",
            "problem": "Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 to bake muffins. She sells the remainder at $2 each. How much does she make per day?",
            "expected": "18"
        },
        {
            "category": "MATH Competition Problem",
            "problem": "What is the remainder when 3^{100} is divided by 5?",
            "expected": "1"
        }
    ]

    print("\n" + "=" * 60)
    print("QUALITATIVE INFERENCE EVALUATION (Model Generated Outputs Only)")
    print("=" * 60)

    results = []
    for tc in test_cases:
        prompt = f"[Q] {tc['problem']} [R]"
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

        out_ids = generate(model, input_ids, max_new_tokens=64, temperature=0.0, device=device, eos_id=tokenizer.eos_id)
        out_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=False)

        reasoning, pred_ans = parse_generated_text(out_text)
        is_corr = calculate_exact_match(pred_ans, tc["expected"])

        print(f"\nCategory: {tc['category']}")
        print(f"Problem: {tc['problem']}")
        print(f"Expected: {tc['expected']}")
        print(f"Model Output:\n{out_text}")
        print(f"Parsed Stated Answer: {pred_ans}")
        print(f"Correct: {'YES' if is_corr else 'NO'}")
        print("-" * 60)

        results.append({
            "category": tc["category"],
            "problem": tc["problem"],
            "expected": tc["expected"],
            "raw_output": out_text,
            "parsed_answer": pred_ans,
            "is_correct": is_corr
        })

    return results

def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--num_test_samples", type=int, default=200)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cpu" if config.get("smoke_test", False) else config["training"].get("device", "cpu")

    tokenizer_path = config["paths"]["tokenizer"]
    tokenizer = MathTokenizer.load(tokenizer_path)

    config["model"]["vocab_size"] = tokenizer.vocab_size
    model_config = MathSLMConfig.from_dict(config["model"])
    model = MathSLM(model_config)

    ckpt_dir = config["paths"]["checkpoints"]
    best_ckpt = os.path.join(ckpt_dir, "model_best.pt")
    final_ckpt = os.path.join(ckpt_dir, "model_final.pt")
    ckpt_path = best_ckpt if os.path.exists(best_ckpt) else final_ckpt

    if not os.path.exists(ckpt_path):
        print("Error: No checkpoint found for evaluation!")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Checkpoint successfully loaded into fresh MathSLM model instance.")

    # Held-out evaluation
    data_dir = config["paths"]["data_processed"]
    ds_dict = load_from_disk(data_dir)
    test_ds = ds_dict["test"]

    print(f"\nEvaluating on held-out test split ({min(args.num_test_samples, len(test_ds))} samples)...")

    test_samples = test_ds.select(range(min(args.num_test_samples, len(test_ds))))
    total_loss = 0.0
    loss_count = 0
    correct = 0

    dataset_stats = {}
    subject_stats = {}

    r_id = tokenizer.r_id
    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id
    max_seq_len = config["model"]["max_seq_len"]

    with torch.no_grad():
        for item in tqdm(test_samples, desc="Computing Test Metrics"):
            q = item["question"]
            r = item["reasoning"]
            a = item["answer"]
            src = item.get("source", "unknown")
            subj = item.get("subject", "general")

            if src not in dataset_stats: dataset_stats[src] = {"total": 0, "correct": 0}
            if subj not in subject_stats: subject_stats[subj] = {"total": 0, "correct": 0}

            dataset_stats[src]["total"] += 1
            subject_stats[subj]["total"] += 1

            # Loss computation
            prompt_full = f"[Q] {q} [R] {r} [A] {a} [EOS]"
            ids = tokenizer.encode(prompt_full)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
                ids[-1] = eos_id
            lbls = list(ids)
            try:
                r_idx = ids.index(r_id)
                for i in range(r_idx): lbls[i] = -100
            except ValueError: pass

            input_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            target_tensor = torch.tensor(lbls, dtype=torch.long).unsqueeze(0).to(device)

            _, loss, _ = model(input_tensor, targets=target_tensor)
            if loss is not None:
                total_loss += loss.item()
                loss_count += 1

            # Generation evaluation
            gen_prompt = f"[Q] {q} [R]"
            gen_input = torch.tensor(tokenizer.encode(gen_prompt), dtype=torch.long).unsqueeze(0).to(device)
            out_ids = generate(model, gen_input, max_new_tokens=64, temperature=0.0, device=device, eos_id=eos_id)
            out_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=False)

            raw_reasoning, pred_ans = parse_generated_text(out_text)
            
            # Robust answer extraction fallback
            if not pred_ans:
                boxed = re.findall(r'\\boxed\{([^}]+)\}', out_text)
                if boxed:
                    pred_ans = boxed[-1]
                else:
                    ans_match = re.search(r'(?:the\s+answer\s+is|answer:?)\s*([^\n\.]+)', out_text, re.IGNORECASE)
                    if ans_match:
                        pred_ans = ans_match.group(1)

            is_corr = calculate_exact_match(pred_ans, a)
            if is_corr:
                correct += 1
                dataset_stats[src]["correct"] += 1
                subject_stats[subj]["correct"] += 1

    avg_test_loss = total_loss / max(1, loss_count)
    perplexity = math.exp(avg_test_loss) if avg_test_loss < 20 else float("inf")
    accuracy = (correct / len(test_samples)) * 100.0

    print("\n" + "=" * 60)
    print(f"HELD-OUT EVALUATION METRICS")
    print("=" * 60)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.4f}")
    print(f"Exact-Answer Accuracy: {accuracy:.2f}% ({correct}/{len(test_samples)})")

    print("\nAccuracy by Dataset:")
    for ds_name, s in dataset_stats.items():
        acc = (s["correct"] / s["total"]) * 100.0 if s["total"] > 0 else 0.0
        print(f" - {ds_name}: {acc:.2f}% ({s['correct']}/{s['total']})")

    print("\nAccuracy by Mathematical Category:")
    for sub_name, s in subject_stats.items():
        acc = (s["correct"] / s["total"]) * 100.0 if s["total"] > 0 else 0.0
        print(f" - {sub_name}: {acc:.2f}% ({s['correct']}/{s['total']})")

    # Run qualitative tests
    qual_results = run_qualitative_tests(model, tokenizer, device)

    # Save complete evaluation summary
    log_dir = config["paths"]["logs"]
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, "final_evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "test_loss": avg_test_loss,
            "perplexity": perplexity,
            "exact_match_accuracy": accuracy,
            "dataset_breakdown": dataset_stats,
            "subject_breakdown": subject_stats,
            "qualitative_test_results": qual_results
        }, f, indent=2)
    print(f"\nFinal evaluation summary saved to {summary_path}")

if __name__ == "__main__":
    main()
