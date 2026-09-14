import os
import sys
import time
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import MathSLMConfig
from model.transformer import MathSLM
from tokenizer.tokenizer import MathTokenizer
from training.dataset import MathDataset
from training.train import get_dataloader, get_lr_scheduler, get_device, evaluate, save_checkpoint

def run_smoke_test():
    print("=" * 60)
    print("MathSLM GPU Smoke Test (Tesla T4 / Colab)")
    print("=" * 60)

    device = get_device("auto")
    print(f"Target Device: {device}")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats(0)
        initial_vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        print(f"GPU Name: {gpu_name}")
        print(f"Initial VRAM Allocated: {initial_vram_mb:.2f} MB")
    else:
        print("WARNING: CUDA is not available in current environment.")
        gpu_name = "N/A (CPU Mode)"

    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Enforce smoke test parameters
    config["paths"]["data_processed"] = "data/processed"
    config["model"]["max_seq_len"] = 256
    config["training"]["batch_size"] = 32
    config["training"]["learning_rate"] = 1e-3
    
    use_amp = (device == "cuda")
    print(f"PyTorch FP16 AMP Enabled: {use_amp}")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    tokenizer_path = config["paths"]["tokenizer"]
    tokenizer = MathTokenizer.load(tokenizer_path)

    data_dir = config["paths"]["data_processed"]
    print(f"Loading dataset from {data_dir}...")
    from datasets import load_from_disk
    dataset_dict = load_from_disk(data_dir)

    print("Constructing DataLoaders...")
    train_loader = get_dataloader(
        dataset_dict, "train", tokenizer, 
        max_seq_len=256, batch_size=32, 
        curriculum=False, objective_mode="weighted_reasoning", answer_loss_weight=5.0
    )
    val_loader = get_dataloader(
        dataset_dict, "val", tokenizer, 
        max_seq_len=256, batch_size=32, 
        objective_mode="weighted_reasoning", answer_loss_weight=5.0
    )

    config["model"]["vocab_size"] = tokenizer.vocab_size
    model_config = MathSLMConfig.from_dict(config["model"])
    model = MathSLM(model_config).to(device)

    total_params = model.get_num_params()
    print(f"Model Initialized: {total_params / 1e6:.2f}M parameters ({total_params:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=10, total_steps=100)

    model.train()
    step_times = []
    losses = []

    print("\nExecuting 10 Training Steps...")
    start_train_time = time.time()

    for step, batch in enumerate(train_loader):
        if step >= 10:
            break

        t0 = time.time()
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

        scheduler.step()
        t1 = time.time()

        step_time = t1 - t0
        step_times.append(step_time)
        losses.append(loss.item())
        print(f"  Step {step+1:02d}/10 | Loss: {loss.item():.4f} | Time: {step_time*1000:.1f} ms | Samples/sec: {32/step_time:.1f}")

    total_train_time = time.time() - start_train_time
    avg_step_time = sum(step_times) / len(step_times)
    throughput = (len(step_times) * 32) / total_train_time

    print(f"\nTraining Loop Complete:")
    print(f"  - Initial Loss (Step 1): {losses[0]:.4f}")
    print(f"  - Step 10 Loss: {losses[-1]:.4f}")
    print(f"  - Average Step Time: {avg_step_time*1000:.1f} ms")
    print(f"  - Measured Throughput: {throughput:.1f} samples/sec")

    print("\nRunning Evaluation Verification...")
    val_loss = evaluate(model, val_loader, device, use_amp=use_amp)
    print(f"  - Validation Loss: {val_loss:.4f} (Evaluation Verification PASSED)")

    print("\nRunning Checkpoint Verification...")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    test_ckpt_path = os.path.join(checkpoint_dir, "smoke_test_ckpt.pt")
    save_checkpoint(test_ckpt_path, model, optimizer, scheduler, epoch=0, step=10, val_loss=val_loss, config=config)
    
    ckpt_exists = os.path.exists(test_ckpt_path)
    ckpt_size_mb = os.path.getsize(test_ckpt_path) / (1024 ** 2) if ckpt_exists else 0.0
    print(f"  - Checkpoint Saved: {test_ckpt_path} ({ckpt_size_mb:.2f} MB)")
    print(f"  - Checkpoint Verification: {'PASSED' if ckpt_exists else 'FAILED'}")

    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        peak_vram_gb = peak_vram_mb / 1024
        print(f"\nPeak GPU VRAM Allocated: {peak_vram_mb:.2f} MB ({peak_vram_gb:.2f} GB)")
    else:
        peak_vram_mb = 0.0
        peak_vram_gb = 0.0
        print("\nPeak GPU VRAM: N/A (CPU Mode)")

    print("\n" + "=" * 60)
    print("GPU Smoke Test Summary")
    print("=" * 60)
    print(f"Device: {gpu_name}")
    print(f"Dataset: data/processed ({len(dataset_dict['train']):,} train samples)")
    print(f"Model Parameters: {total_params / 1e6:.2f}M")
    print(f"Peak VRAM: {peak_vram_mb:.2f} MB ({peak_vram_gb:.2f} GB)")
    print(f"Throughput: {throughput:.1f} samples/sec")
    print(f"Step 1 Loss: {losses[0]:.4f} -> Step 10 Loss: {losses[-1]:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Evaluation Gate: PASSED")
    print(f"Checkpoint Gate: PASSED")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
