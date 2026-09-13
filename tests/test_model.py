import torch
import unittest
from model.config import MathSLMConfig
from model.transformer import MathSLM

class TestMathSLM(unittest.TestCase):
    def setUp(self):
        self.config = MathSLMConfig(
            vocab_size=500,
            max_seq_len=64,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            use_rope=True
        )
        self.model = MathSLM(self.config)
        self.model.eval()

    def test_output_shape(self):
        input_ids = torch.randint(0, 500, (2, 16))
        logits, loss, _ = self.model(input_ids)

        self.assertEqual(logits.shape, (2, 16, 500))
        self.assertIsNone(loss)

    def test_causal_masking(self):
        input_ids_1 = torch.tensor([[1, 2, 3, 4, 5]])
        input_ids_2 = torch.tensor([[1, 2, 3, 99, 99]])

        with torch.no_grad():
            logits_1, _, _ = self.model(input_ids_1)
            logits_2, _, _ = self.model(input_ids_2)

        self.assertTrue(torch.allclose(logits_1[0, :3, :], logits_2[0, :3, :], atol=1e-4))
        self.assertFalse(torch.allclose(logits_1[0, 3, :], logits_2[0, 3, :], atol=1e-4))

    def test_loss_computation(self):
        input_ids = torch.randint(0, 500, (2, 16))
        targets = input_ids.clone()
        targets[:, :5] = -100

        logits, loss, _ = self.model(input_ids, targets=targets)
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0.0)

    def test_parameter_count(self):
        n_params = self.model.get_num_params()
        self.assertTrue(n_params > 0)
        print(f"Test model parameter count: {n_params}")

if __name__ == "__main__":
    unittest.main()
