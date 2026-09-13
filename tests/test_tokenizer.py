import unittest
import os
import tempfile
from tokenizers import Tokenizer, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Whitespace, Digits, Split

from tokenizer.tokenizer import MathTokenizer

class TestMathTokenizer(unittest.TestCase):
    def setUp(self):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Sequence([
            Whitespace(),
            Digits(individual_digits=True),
            Split(Regex(r"\\[a-zA-Z]+"), behavior="isolated")
        ])
        tokenizer.decoder = decoders.BPEDecoder()

        special_tokens = ["[PAD]", "[UNK]", "[EOS]", "[Q]", "[R]", "[A]"]
        trainer = BpeTrainer(vocab_size=1000, special_tokens=special_tokens)

        corpus = [
            "[Q] Solve 2x + 7 = 19 [R] 2x = 12, so x = 6 [A] 6 [EOS]",
            "[Q] What is \\frac{a}{b} + \\sqrt{x}? [R] Simplify expression. [A] \\frac{a}{b} [EOS]"
        ]
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        self.math_tokenizer = MathTokenizer(tokenizer)

    def test_special_tokens(self):
        self.assertIsNotNone(self.math_tokenizer.pad_id)
        self.assertIsNotNone(self.math_tokenizer.eos_id)
        self.assertIsNotNone(self.math_tokenizer.q_id)
        self.assertIsNotNone(self.math_tokenizer.r_id)
        self.assertIsNotNone(self.math_tokenizer.a_id)

    def test_encode_decode_roundtrip(self):
        text = "[Q] Solve 2x + 7 = 19 [R] 2x = 12 [A] 6 [EOS]"
        ids = self.math_tokenizer.encode(text)
        decoded = self.math_tokenizer.decode(ids, skip_special_tokens=False)
        self.assertIn("Solve", decoded)
        self.assertIn("19", decoded)

    def test_serialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "tokenizer.json")
            self.math_tokenizer.save(save_path)
            loaded_tok = MathTokenizer.load(save_path)
            self.assertEqual(self.math_tokenizer.vocab_size, loaded_tok.vocab_size)

if __name__ == "__main__":
    unittest.main()
