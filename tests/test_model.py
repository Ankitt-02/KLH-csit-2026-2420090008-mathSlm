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

    def test_answer_parsing_formats(self):
        from model.generation import parse_generated_text
        from verification.symbolic_verifier import SymbolicVerifier
        verifier = SymbolicVerifier()

        # Integer
        _, ans1 = parse_generated_text("[Q] Test [R] Work [A] 42 [EOS]")
        self.assertEqual(ans1, "42")
        self.assertTrue(verifier.verify("", ans1, "42")[0])

        # Decimal
        _, ans2 = parse_generated_text("[Q] Test [R] Work [A] 3.14 [EOS]")
        self.assertEqual(ans2, "3.14")
        self.assertTrue(verifier.verify("", ans2, "3.14")[0])

        # Negative
        _, ans3 = parse_generated_text("[Q] Test [R] Work [A] -17 [EOS]")
        self.assertEqual(ans3, "-17")
        self.assertTrue(verifier.verify("", ans3, "-17")[0])

        # Fraction
        _, ans4 = parse_generated_text("[Q] Test [R] Work [A] 3/4 [EOS]")
        self.assertEqual(ans4, "3/4")

        # LaTeX Boxed
        _, ans5 = parse_generated_text("[Q] Test [R] Work [A] \\boxed{100} [EOS]")
        self.assertEqual(ans5, "100")
        self.assertTrue(verifier.verify("", ans5, "100")[0])

    def test_mathqa_extraction(self):
        from data.prepare_data import parse_mathqa_options, extract_mathqa_answer

        options_str = "a ) 14 , b ) 15 , c ) 16 , d ) 17 , e ) 18"
        parsed = parse_mathqa_options(options_str)
        self.assertEqual(parsed.get("a"), "14")
        self.assertEqual(parsed.get("b"), "15")

        item1 = {"correct": "b", "options": options_str, "Rationale": "some explanation"}
        ans1 = extract_mathqa_answer(item1)
        self.assertEqual(ans1, "15")

        item2 = {"correct": "z", "options": "", "Rationale": "the answer : 99."}
        ans2 = extract_mathqa_answer(item2)
        self.assertEqual(ans2, "99")

if __name__ == "__main__":
    unittest.main()
