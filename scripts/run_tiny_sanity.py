import sys
import os
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import MathSLMConfig
from model.transformer import MathSLM
from tokenizer.tokenizer import MathTokenizer
from training.dataset import MathDataset
from training.train import collate_fn_batch, get_device
from model.generation import generate

# Configuration Constants
DATASET_SIZE = 200
EPOCHS = 20
BATCH_SIZE = 4
MAX_SEQ_LEN = 64
OBJECTIVE_MODE = "direct"

def check_exact_match(pred, expected):
    pred = pred.strip()
    expected = expected.strip()
    if pred == expected:
        return True
    try:
        return abs(float(pred) - float(expected)) < 1e-4
    except Exception:
        return False

def run_tiny_sanity():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    dataset_size = DATASET_SIZE
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    max_seq_len = MAX_SEQ_LEN
    objective_mode = OBJECTIVE_MODE

    batches_per_epoch = math.ceil(dataset_size / batch_size)
    expected_optimizer_steps = batches_per_epoch * epochs

    print("=" * 60, flush=True)
    print(f"RUNNING TINY SANITY TEST ({dataset_size} samples, {epochs} epochs, batch_size={batch_size}, 7.34M model)", flush=True)
    print(f"Batches/Epoch: {batches_per_epoch} | Expected Optimizer Steps: {expected_optimizer_steps}", flush=True)
    print("=" * 60, flush=True)

    assert expected_optimizer_steps == 1000, f"Step budget mismatch: Expected 1000 optimizer steps, got {expected_optimizer_steps}"

    torch.set_num_threads(4)
    device = get_device("auto")

    tokenizer = MathTokenizer.load("tokenizer/math_tokenizer.json")
    ds_dict = load_from_disk("data/processed_synthetic")
    tiny_ds = ds_dict["train"].select(range(dataset_size))

    train_dataset = MathDataset(tiny_ds, tokenizer, max_seq_len, objective_mode=objective_mode)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_batch)

    model_config = MathSLMConfig.from_dict({
        "vocab_size": tokenizer.vocab_size,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "max_seq_len": max_seq_len,
        "ffn_dim": 1024,
        "dropout": 0.0
    })
    model = MathSLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    initial_loss = None
    final_loss = None

    print(f"Training 7.34M model on {len(tiny_ds)} examples for {epochs} epochs (batch_size={batch_size}, total steps={expected_optimizer_steps})...", flush=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, loss, _ = model(input_ids, targets=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        if initial_loss is None:
            initial_loss = avg_loss
        final_loss = avg_loss

        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f}", flush=True)

    loss_drop_pct = ((initial_loss - final_loss) / initial_loss) * 100.0
    print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f} (Drop: {loss_drop_pct:.2f}%)", flush=True)

    eval_samples = tiny_ds.select(range(50))
    model.eval()
    correct = 0
    failures = 0

    print(f"\nEvaluating exact-answer accuracy on {len(eval_samples)} training samples (memorization gate)...", flush=True)
    with torch.no_grad():
        for i, item in enumerate(eval_samples):
            q = item["question"]
            expected = item["answer"]
            gen_prompt = f"[Q] {q} [A] "
            gen_input = torch.tensor(tokenizer.encode(gen_prompt), dtype=torch.long).unsqueeze(0).to(device)

            out_ids = generate(model, gen_input, max_new_tokens=10, temperature=0.0, device=device, eos_id=tokenizer.eos_id)
            out_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=False)

            pred_ans = ""
            if "[A]" in out_text:
                pred_ans = out_text.split("[A]")[1].replace("[EOS]", "").strip()
                if " " in pred_ans:
                    pred_ans = pred_ans.split()[0]
            else:
                failures += 1

            is_corr = check_exact_match(pred_ans, expected)
            if is_corr:
                correct += 1

            if (i + 1) % 10 == 0 or i == len(eval_samples) - 1:
                print(f"Sample {i+1:02d}/{len(eval_samples)} | Pred: '{pred_ans}' | Expected: '{expected}' | Correct: {'YES' if is_corr else 'NO'}", flush=True)

    accuracy = (correct / len(eval_samples)) * 100.0
    print(f"\nTiny Sanity Exact-Answer Accuracy: {accuracy:.2f}% ({correct}/{len(eval_samples)})", flush=True)
    print(f"Generation Failure Count: {failures}", flush=True)

    if loss_drop_pct >= 80.0 and accuracy >= 80.0:
        print("\n" + "=" * 60, flush=True)
        print("✅ TINY SANITY TEST PASSED! PIPELINE AND MODEL CAN FIT SMALL ARITHMETIC DATASET.", flush=True)
        print("=" * 60, flush=True)
        return 0
    else:
        print("\n" + "=" * 60, flush=True)
        print(f"❌ TINY SANITY TEST FAILED! Loss Drop: {loss_drop_pct:.1f}%, Acc: {accuracy:.1f}%. STOPPING DIAGNOSTIC MATRIX.", flush=True)
        print("=" * 60, flush=True)
        return 1

if __name__ == "__main__":
    sys.exit(run_tiny_sanity())
