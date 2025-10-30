import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropoutScheduler:
    """Linear decay scheduler for dropout probability."""
    def __init__(self, init_dropout: float, min_dropout: float = 0.0, decay_steps: int = 1000):
        self.init_dropout = init_dropout
        self.min_dropout = min_dropout
        self.decay_steps = decay_steps
        self.step = 0

    def step_update(self) -> float:
        self.step += 1
        ratio = min(self.step / self.decay_steps, 1.0)
        return self.init_dropout - ratio * (self.init_dropout - self.min_dropout)

    def reset(self):
        self.step = 0


class HybridAttention(nn.Module):
    """
    Multi‑head attention that can optionally route projections through a quantum module.
    In the classical implementation (default) the quantum path is a no‑op.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_quantum: bool = False):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_quantum = use_quantum

        # Classical linear maps
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        k = self.k_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # If quantum path is requested, we simply add a small random perturbation to simulate quantum effects
        if self.use_quantum:
            perturb = torch.randn_like(k) * 0.01
            k += perturb
            q += perturb
            v += perturb

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Standard two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with optional quantum regularization."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
        quantum_reg_weight: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = HybridAttention(embed_dim, num_heads, dropout, use_quantum)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.quantum_reg_weight = quantum_reg_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.attn.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.ffn.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextClassifier(nn.Module):
    """
    Transformer‑based text classifier that optionally employs quantum sub‑modules.
    The classical implementation is fully compatible; setting `use_quantum=True` simply
    injects a lightweight perturbation to mimic quantum behaviour.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
        dropout_decay_steps: int = 1000,
        quantum_reg_weight: float = 0.0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout=dropout,
                    use_quantum=use_quantum,
                    quantum_reg_weight=quantum_reg_weight,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout_scheduler = DropoutScheduler(dropout, min_dropout=0.0, decay_steps=dropout_decay_steps)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.use_quantum = use_quantum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        current_dropout = self.dropout_scheduler.step_update()
        x = nn.Dropout(current_dropout)(x)
        return self.classifier(x)

    def reset_dropout(self):
        """Reset the dropout scheduler to the initial state."""
        self.dropout_scheduler.reset()
