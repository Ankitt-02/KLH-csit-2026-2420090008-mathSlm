from dataclasses import dataclass
from typing import Optional

@dataclass
class MathSLMConfig:
    vocab_size: int = 16384
    max_seq_len: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    ffn_dim: Optional[int] = None
    dropout: float = 0.1
    use_rope: bool = True

    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.n_embd

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered_dict)
