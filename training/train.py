import argparse
import math
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
from datetime import datetime
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.config import MathSLMConfig
from model.transformer import MathSLM
from tokenizer.tokenizer import MathTokenizer
from training.dataset import MathDataset
from training.curriculum import CurriculumBatchSampler

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def collate_fn_batch(batch):
    if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], dict):
        item = batch[0]
        input_ids = item["input_ids"]
        labels = item["labels"]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        res = {"input_ids": input_ids, "labels": labels}
        if "loss_weights" in item:
            lw = item["loss_weights"]
            if lw.dim() == 1:
                lw = lw.unsqueeze(0)
            res["loss_weights"] = lw
        return res
    elif isinstance(batch, list) and isinstance(batch[0], dict):
        res = {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch])
        }
        if "loss_weights" in batch[0]:
            res["loss_weights"] = torch.stack([x["loss_weights"] for x in batch])
        return res
    return batch

def get_dataloader(dataset_dict, split, tokenizer, max_seq_len, batch_size, curriculum=False, objective_mode="reasoning", answer_loss_weight=5.0):
    hf_ds = dataset_dict[split]
    dataset = MathDataset(hf_ds, tokenizer, max_seq_len, objective_mode=objective_mode, answer_loss_weight=answer_loss_weight)
    if curriculum and split == "train":
        batch_sampler = CurriculumBatchSampler(hf_ds, batch_size)
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn_batch)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate_fn_batch)

def get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_device(device_setting="auto"):
    if device_setting == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_setting

def save_checkpoint(path, model, optimizer, scheduler, epoch, step, val_loss, config):
    state = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
        "config": config,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
    }
    torch.save(state, path)

def evaluate(model, val_loader, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_weights = batch.get("loss_weights", None)
            if loss_weights is not None:
                loss_weights = loss_weights.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss, _ = model(input_ids, targets=labels, loss_weights=loss_weights)

            if loss is not None:
                total_loss += loss.item()
                count += 1
    model.train()
    avg_loss = total_loss / max(1, count)
    return avg_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--objective_mode", type=str, default="reasoning", choices=["direct", "reasoning", "weighted_reasoning"])
    parser.add_argument("--answer_loss_weight", type=float, default=5.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["true", "false"])
    args = parser.parse_args()

    config = load_config(args.config)
    smoke_test = config.get("smoke_test", False)

    device_setting = args.device if args.device else config["training"].get("device", "auto")
    if smoke_test:
        print("Running in SMOKE TEST mode on CPU.")
        device_setting = "cpu"

    device = get_device(device_setting)
    print(f"Using target training device: {device}")
    if device == "cuda":
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")

    if args.mixed_precision is not None:
        mixed_precision_setting = (args.mixed_precision.lower() == "true")
    else:
        mixed_precision_setting = config["training"].get("mixed_precision", (device == "cuda"))

    use_amp = (device == "cuda") and mixed_precision_setting
    print(f"PyTorch AMP Mixed Precision Enabled: {use_amp}")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    tokenizer_path = config["paths"]["tokenizer"]
    try:
        tokenizer = MathTokenizer.load(tokenizer_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        return

    data_dir = config["paths"]["data_processed"]
    print(f"Loading datasets from {data_dir}...")
    try:
        dataset_dict = load_from_disk(data_dir)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    max_seq_len = config["model"]["max_seq_len"]
    batch_size = config["training"]["batch_size"]

    print(f"Building DataLoaders (Objective: {args.objective_mode}, Weight: {args.answer_loss_weight})...")
    train_loader = get_dataloader(dataset_dict, "train", tokenizer, max_seq_len, batch_size, curriculum=True, objective_mode=args.objective_mode, answer_loss_weight=args.answer_loss_weight)
    val_loader = get_dataloader(dataset_dict, "val", tokenizer, max_seq_len, batch_size, objective_mode=args.objective_mode, answer_loss_weight=args.answer_loss_weight)

    config["model"]["vocab_size"] = tokenizer.vocab_size
    model_config = MathSLMConfig.from_dict(config["model"])
    model = MathSLM(model_config)
    model.to(device)

    total_params = model.get_num_params()
    non_emb_params = model.get_num_params(non_embedding=True)
    print(f"Model Initialized:")
    print(f" - Layers: {model_config.n_layer}, Heads: {model_config.n_head}, Hidden Dim: {model_config.n_embd}, FFN Dim: {model_config.ffn_dim}")
    print(f" - Total Parameters: {total_params / 1e6:.2f}M ({total_params:,} parameters)")
    print(f" - Non-Embedding Parameters: {non_emb_params / 1e6:.2f}M ({non_emb_params:,} parameters)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    epochs = config["training"]["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    start_step = 0
    start_epoch = 0
    checkpoint_dir = config["paths"]["checkpoints"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint.get("step", 0)
        start_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
        if "rng_state" in checkpoint and checkpoint["rng_state"]:
            rng = checkpoint["rng_state"]
            if "torch" in rng and rng["torch"] is not None:
                torch.set_rng_state(rng["torch"])
            if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"Resumed training from checkpoint {args.resume} at step {start_step} (Epoch {start_epoch+1}).")

    log_dir = config["paths"]["logs"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    logs = []

    eval_steps = config["training"]["eval_steps"]
    save_steps = config["training"]["save_steps"]

    global_step = start_step
    skip_batches = start_step % steps_per_epoch if steps_per_epoch > 0 else 0
    best_val_loss = float("inf")

    model.train()
    print(f"Starting training for {epochs} epochs ({total_steps} total steps, starting epoch {start_epoch+1}, skip {skip_batches} batches)...")

    for epoch in range(start_epoch, epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            if epoch == start_epoch and step < skip_batches:
                continue

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
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()

            scheduler.step()

            global_step += 1
            curr_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{curr_lr:.2e}"})

            if global_step % eval_steps == 0 or global_step == total_steps:
                val_loss = evaluate(model, val_loader, device, use_amp=use_amp)
                perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")

                log_entry = {
                    "step": global_step,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "perplexity": perplexity,
                    "lr": curr_lr,
                    "grad_norm": grad_norm
                }
                logs.append(log_entry)
                pd.DataFrame(logs).to_csv(log_file, index=False)

                print(f"\n[Step {global_step}/{total_steps}] Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | PPL: {perplexity:.4f} | LR: {curr_lr:.2e}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(checkpoint_dir, "model_best.pt")
                    save_checkpoint(best_path, model, optimizer, scheduler, epoch, global_step, val_loss, config)
                    print(f"New best model saved to {best_path}")

            if global_step % save_steps == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"model_step_{global_step}.pt")
                save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, global_step, loss.item(), config)
                print(f"Saved periodic checkpoint to {ckpt_path}")

            # Save latest checkpoint after every step for Google Colab crash recovery
            latest_path = os.path.join(checkpoint_dir, "model_latest.pt")
            save_checkpoint(latest_path, model, optimizer, scheduler, epoch, global_step, loss.item(), config)

    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    save_checkpoint(final_path, model, optimizer, scheduler, epochs, global_step, best_val_loss, config)
    print(f"Training complete! Final checkpoint saved to {final_path}")

if __name__ == "__main__":
    main()
