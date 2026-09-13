import os
import torch
from tokenizers import Tokenizer

class MathTokenizer:
    """
    Wrapper class around HuggingFace tokenizers.Tokenizer with convenience methods
    for mathematical token encoding, decoding, padding, and special token management.
    """
    def __init__(self, tokenizer_path_or_obj):
        if isinstance(tokenizer_path_or_obj, str):
            self.tokenizer = Tokenizer.from_file(tokenizer_path_or_obj)
        else:
            self.tokenizer = tokenizer_path_or_obj

        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.eos_token = "[EOS]"
        self.q_token = "[Q]"
        self.r_token = "[R]"
        self.a_token = "[A]"

        self.pad_id = self.token_to_id(self.pad_token)
        self.unk_id = self.token_to_id(self.unk_token)
        self.eos_id = self.token_to_id(self.eos_token)
        self.q_id = self.token_to_id(self.q_token)
        self.r_id = self.token_to_id(self.r_token)
        self.a_id = self.token_to_id(self.a_token)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        tid = self.tokenizer.token_to_id(token)
        if tid is None:
            return self.tokenizer.token_to_id("[UNK]")
        return tid

    def id_to_token(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(token_id)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str):
        return cls(path)
