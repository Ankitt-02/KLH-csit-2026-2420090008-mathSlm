import sys
import os
import math
import time
import json
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import MathSLMConfig
from model.transformer import MathSLM
from tokenizer.tokenizer import MathTokenizer
from training.dataset import MathDataset
from training.train import collate_fn_batch, get_device, save_checkpoint
from model.generation import generate
from evaluation.metrics import calculate_exact_match

import random
import numpy as np

def run_experiment(exp_name, exp_id, model_params, objective_mode, epochs, dataset_path="data/processed_synthetic", device_setting="auto", mixed_precision=True):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("\n" + "=" * 70, flush=True)
    print(f"STARTING EXPERIMENT: {exp_name} ({exp_id})", flush=True)
    print(f"Model: {model_params['n_layer']}L / {model_params['n_head']}H / {model_params['n_embd']}D | Objective: {objective_mode} | Epochs: {epochs}", flush=True)
    print("=" * 70, flush=True)

    device = get_device(device_setting)
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.memory_allocated(0)/1e6:.2f}MB", flush=True)

    use_amp = (device == "cuda") and mixed_precision
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    tokenizer = MathTokenizer.load("tokenizer/math_tokenizer.json")
    ds_dict = load_from_disk(dataset_path)
    train_ds = ds_dict["train"]
    test_ds = ds_dict["test"]

    max_seq_len = 128
    batch_size = 32 if device == "cuda" else 16

    train_dataset = MathDataset(train_ds, tokenizer, max_seq_len, objective_mode=objective_mode)
    test_dataset = MathDataset(test_ds, tokenizer, max_seq_len, objective_mode=objective_mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_batch)

    model_config = MathSLMConfig.from_dict({
        "vocab_size": tokenizer.vocab_size,
        "n_layer": model_params["n_layer"],
        "n_head": model_params["n_head"],
        "n_embd": model_params["n_embd"],
        "max_seq_len": max_seq_len,
        "ffn_dim": model_params.get("ffn_dim", 4 * model_params["n_embd"])
    })

    model = MathSLM(model_config).to(device)
    total_params = model.get_num_params()
    print(f"Instantiated Model Parameters: {total_params:,} ({total_params/1e6:.2f}M)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Use isolated directory for each experiment: checkpoints/v2_exp_a, etc.
    ckpt_dir = os.path.join("checkpoints", exp_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training Loop
    start_time = time.time()
    best_loss = float("inf")
    last_train_loss = 0.0

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_weights = batch.get("loss_weights", None)
            if loss_weights is not None:
                loss_weights = loss_weights.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, loss, _ = model(input_ids, targets=labels, loss_weights=loss_weights)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        last_train_loss = avg_loss
        print(f"Epoch {ep+1:02d}/{epochs} | Train Loss: {avg_loss:.4f}", flush=True)

        # Save continuous latest checkpoint
        save_checkpoint(os.path.join(ckpt_dir, "model_latest.pt"), model, optimizer, None, ep, ep * batches, avg_loss, model_config.__dict__)

        # Save best model checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(os.path.join(ckpt_dir, "model_best.pt"), model, optimizer, None, ep, ep * batches, avg_loss, model_config.__dict__)

    training_time = time.time() - start_time

    # Evaluate Test Loss & Perplexity
    model.eval()
    test_loss = 0.0
    t_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss, _ = model(input_ids, targets=labels)
            if loss is not None:
                test_loss += loss.item()
                t_batches += 1
    avg_test_loss = test_loss / max(1, t_batches)
    perplexity = math.exp(avg_test_loss) if avg_test_loss < 20 else float("inf")

    # Evaluate Training Accuracy (100 samples) and Held-out Test Accuracy (200 samples)
    def evaluate_exact_accuracy(samples, desc=""):
        correct = 0
        failures = 0
        by_subject = {}
        for item in samples:
            q = item["question"]
            expected = item["answer"]
            subj = item.get("subject", "general")
            if subj not in by_subject:
                by_subject[subj] = {"total": 0, "correct": 0}
            by_subject[subj]["total"] += 1

            gen_prompt = f"[Q] {q} [A] " if objective_mode == "direct" else f"[Q] {q} [R] "
            gen_input = torch.tensor(tokenizer.encode(gen_prompt), dtype=torch.long).unsqueeze(0).to(device)

            out_ids = generate(model, gen_input, max_new_tokens=32, temperature=0.0, device=device, eos_id=tokenizer.eos_id)
            out_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=False)

            pred_ans = ""
            if "[A]" in out_text:
                pred_ans = out_text.split("[A]")[1].replace("[EOS]", "").strip()
                if " " in pred_ans:
                    pred_ans = pred_ans.split()[0]
            elif "\\boxed{" in out_text:
                pred_ans = out_text.split("\\boxed{")[1].split("}")[0].strip()
            else:
                failures += 1

            is_c = calculate_exact_match(pred_ans, expected)
            if is_c:
                correct += 1
                by_subject[subj]["correct"] += 1

        acc = (correct / len(samples)) * 100.0
        return acc, correct, len(samples) - correct, failures, by_subject

    print(f"\nEvaluating exact-answer accuracy...", flush=True)
    train_acc, train_corr, train_inc, train_fail, _ = evaluate_exact_accuracy(train_ds.select(range(100)), "Train")
    test_acc, test_corr, test_inc, test_fail, subj_breakdown = evaluate_exact_accuracy(test_ds.select(range(200)), "Held-out Test")

    print(f"\nResults for {exp_name} ({exp_id}):", flush=True)
    print(f" - Train Loss: {last_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test PPL: {perplexity:.4f}", flush=True)
    print(f" - Train Accuracy: {train_acc:.2f}% ({train_corr}/100)", flush=True)
    print(f" - Held-out Test Accuracy: {test_acc:.2f}% ({test_corr}/200)", flush=True)
    print(f" - Generation Failures: {test_fail}", flush=True)
    print(" - Accuracy by Subject:", flush=True)
    for s_name, s_data in subj_breakdown.items():
        s_acc = (s_data["correct"] / s_data["total"]) * 100.0 if s_data["total"] > 0 else 0.0
        print(f"    * {s_name}: {s_acc:.1f}% ({s_data['correct']}/{s_data['total']})", flush=True)

    result = {
        "exp_name": exp_name,
        "exp_id": exp_id,
        "model_params": model_params,
        "total_params": total_params,
        "objective_mode": objective_mode,
        "epochs": epochs,
        "train_loss": last_train_loss,
        "test_loss": avg_test_loss,
        "perplexity": perplexity,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_correct": train_corr,
        "train_incorrect": train_inc,
        "test_correct": test_corr,
        "test_incorrect": test_inc,
        "generation_failures": test_fail,
        "subject_breakdown": subj_breakdown,
        "training_time_sec": training_time
    }
    return result

def main():
    device_setting = sys.argv[1] if len(sys.argv) > 1 else "auto"
    
    experiments = [
        {
            "exp_name": "Exp-A (7.34M Direct 5ep)",
            "exp_id": "v2_exp_a",
            "model_params": {"n_layer": 4, "n_head": 4, "n_embd": 256, "ffn_dim": 1024},
            "objective_mode": "direct",
            "epochs": 5
        },
        {
            "exp_name": "Exp-B (30M Direct 5ep)",
            "exp_id": "v2_exp_b",
            "model_params": {"n_layer": 8, "n_head": 8, "n_embd": 512, "ffn_dim": 2048},
            "objective_mode": "direct",
            "epochs": 5
        },
        {
            "exp_name": "Exp-C (7.34M Reasoning 5ep)",
            "exp_id": "v2_exp_c",
            "model_params": {"n_layer": 4, "n_head": 4, "n_embd": 256, "ffn_dim": 1024},
            "objective_mode": "reasoning",
            "epochs": 5
        },
        {
            "exp_name": "Exp-D (7.34M Reasoning 1ep)",
            "exp_id": "v2_exp_d",
            "model_params": {"n_layer": 4, "n_head": 4, "n_embd": 256, "ffn_dim": 1024},
            "objective_mode": "reasoning",
            "epochs": 1
        }
    ]

    results = []
    drive_dir = os.environ.get("DRIVE_DIR", "/content/drive/MyDrive/MathSLM_v2")

    for exp in experiments:
        res = run_experiment(
            exp["exp_name"],
            exp["exp_id"],
            exp["model_params"],
            exp["objective_mode"],
            exp["epochs"],
            device_setting=device_setting
        )
        results.append(res)

        # Save local summary JSON after each experiment
        os.makedirs("logs", exist_ok=True)
        out_file = "logs/v2_diagnostic_experiments_summary.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

        # Persistence to Google Drive immediately after each experiment completes
        if os.path.exists(drive_dir):
            try:
                local_ckpt = os.path.join("checkpoints", exp["exp_id"])
                drive_ckpt = os.path.join(drive_dir, "checkpoints", exp["exp_id"])
                drive_logs = os.path.join(drive_dir, "logs")

                os.makedirs(drive_ckpt, exist_ok=True)
                os.makedirs(drive_logs, exist_ok=True)

                shutil.copytree(local_ckpt, drive_ckpt, dirs_exist_ok=True)
                shutil.copy(out_file, os.path.join(drive_logs, "v2_diagnostic_experiments_summary.json"))

                print(f"✅ Immediate Google Drive Sync: Persisted '{exp['exp_id']}' checkpoints to {drive_ckpt}", flush=True)
            except Exception as e:
                print(f"⚠️ Warning: Failed to sync {exp['exp_id']} to Google Drive: {e}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"ALL 4 CONTROLLED EXPERIMENTS COMPLETED. SUMMARY SAVED TO {out_file}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
