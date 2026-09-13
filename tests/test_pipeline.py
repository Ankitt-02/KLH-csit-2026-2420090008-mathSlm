import os
import tempfile
import torch
import unittest

from model.config import MathSLMConfig
from model.transformer import MathSLM
from model.generation import generate, parse_generated_text

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.config = MathSLMConfig(
            vocab_size=200,
            max_seq_len=32,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0
        )
        self.model = MathSLM(self.config)

    def test_optimizer_step_and_gradient_flow(self):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, 200, (2, 16))
        targets = input_ids.clone()
        targets[:, :4] = -100

        logits, loss_initial, _ = self.model(input_ids, targets=targets)
        initial_val = loss_initial.item()

        optimizer.zero_grad()
        loss_initial.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient!")
                self.assertTrue(torch.norm(param.grad) >= 0.0)

        optimizer.step()

        _, loss_after, _ = self.model(input_ids, targets=targets)
        self.assertTrue(loss_after.item() <= initial_val, "Loss did not decrease after optimization step!")

    def test_checkpoint_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model_test.pt")
            torch.save(self.model.state_dict(), ckpt_path)

            new_model = MathSLM(self.config)
            new_model.load_state_dict(torch.load(ckpt_path))

            input_ids = torch.randint(0, 200, (1, 10))
            with torch.no_grad():
                l1, _, _ = self.model(input_ids)
                l2, _, _ = new_model(input_ids)

            self.assertTrue(torch.allclose(l1, l2, atol=1e-5))

    def test_generation_and_parsing(self):
        self.model.eval()
        input_ids = torch.tensor([[10, 20, 30]])
        out_ids = generate(self.model, input_ids, max_new_tokens=10, temperature=0.0)

        self.assertEqual(out_ids.shape[0], 1)
        self.assertEqual(out_ids.shape[1], 13)

        text_sample = "[Q] Solve 2+2 [R] 2+2=4 [A] 4 [EOS]"
        reasoning, answer = parse_generated_text(text_sample)
        self.assertEqual(reasoning, "2+2=4")
        self.assertEqual(answer, "4")

if __name__ == "__main__":
    unittest.main()
