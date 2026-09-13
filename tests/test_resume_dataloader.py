import os
import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __init__(self, size=20):
        self.data = [torch.tensor([float(i)], dtype=torch.float32) for i in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

class TestResumeDataloader(unittest.TestCase):
    def test_resume_fast_forward_and_optimizer_state(self):
        dataset = DummyDataset(size=20)
        batch_size = 4
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        steps_per_epoch = len(loader)  # 5 batches

        model = DummyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # Run 5 steps (1 full epoch)
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            out = model(batch)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Save checkpoint at step 5
        ckpt_path = "test_ckpt.pt"
        torch.save({
            "step": 5,
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": {"torch": torch.get_rng_state()}
        }, ckpt_path)

        # Instantiate fresh model, optimizer, scheduler
        resumed_model = DummyModel()
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
        resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=5, gamma=0.5)

        # Load checkpoint
        ckpt = torch.load(ckpt_path)
        resumed_model.load_state_dict(ckpt["model_state_dict"])
        resumed_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        resumed_scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        skip_batches = start_step % steps_per_epoch  # 0 batches skip for epoch 2

        epochs = 2
        processed_batches = []
        global_step = start_step

        for epoch in range(start_epoch, epochs):
            for step, batch in enumerate(loader):
                if epoch == start_epoch and step < skip_batches:
                    continue
                processed_batches.append((epoch, step, batch.tolist()))
                global_step += 1

        # Check total steps ran from step 5 to 10 (5 steps)
        self.assertEqual(global_step, 10)
        self.assertEqual(len(processed_batches), 5)
        self.assertEqual(processed_batches[0][0], 1)  # Epoch 2 (0-indexed 1)
        self.assertEqual(processed_batches[0][1], 0)  # Batch 0

        # Clean up
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        print("Resume Unit Test (Dataloader, Model, Optimizer, Scheduler) PASSED!")

if __name__ == "__main__":
    unittest.main()
