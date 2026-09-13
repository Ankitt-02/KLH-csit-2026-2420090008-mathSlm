import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.config import MathSLMConfig

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len: int, past_length: int = 0):
        # q & k shape: (B, n_head, T, head_dim)
        # cos & sin shape: (1, 1, T, head_dim)
        cos = self.cos_cached[past_length : past_length + seq_len, : self.dim].unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[past_length : past_length + seq_len, : self.dim].unsqueeze(0).unsqueeze(1)
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, config: MathSLMConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_rope = config.use_rope

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=config.max_seq_len)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
            persistent=False
        )

    def forward(self, x, use_cache=False, past_key_value=None):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_length = 0 if past_key_value is None else past_key_value[0].size(-2)

        if self.use_rope:
            q, k = self.rope(q, k, seq_len=T, past_length=past_length)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present = (k, v) if use_cache else None
        full_T = k.size(-2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, full_T - T : full_T, :full_T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, present

class MLP(nn.Module):
    def __init__(self, config: MathSLMConfig):
        super().__init__()
        ffn_dim = config.ffn_dim if config.ffn_dim else 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, ffn_dim, bias=False)
        self.c_proj = nn.Linear(ffn_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: MathSLMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, use_cache=False, past_key_value=None):
        attn_out, present = self.attn(self.ln_1(x), use_cache=use_cache, past_key_value=past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present

class MathSLM(nn.Module):
    def __init__(self, config: MathSLMConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        if not config.use_rope:
            self.transformer["wpe"] = nn.Embedding(config.max_seq_len, config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, loss_weights=None, use_cache=False, past_key_values=None):
        device = input_ids.device
        b, t = input_ids.size()

        if past_key_values is None:
            past_key_values = [None] * len(self.transformer.h)

        tok_emb = self.transformer.wte(input_ids)
        if not self.config.use_rope:
            past_length = 0 if past_key_values[0] is None else past_key_values[0][0].size(-2)
            pos = torch.arange(past_length, t + past_length, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        presents = () if use_cache else None

        for block, past in zip(self.transformer.h, past_key_values):
            x, present = block(x, use_cache=use_cache, past_key_value=past)
            if use_cache:
                presents = presents + (present,)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            if loss_weights is not None:
                shift_weights = loss_weights[..., 1:].contiguous().view(-1)
                raw_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction='none')
                valid_mask = (shift_labels.view(-1) != -100)
                weighted_loss = raw_loss * shift_weights
                loss = weighted_loss[valid_mask].sum() / torch.clamp(shift_weights[valid_mask].sum(), min=1.0)
            else:
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        return logits, loss, presents

    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params
