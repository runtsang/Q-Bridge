"""Hybrid quantum regression model: quantum encoder + classical transformer."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torchquantum as tq
import torch.nn.functional as F
import torch.utils.data

def generate_superposition_data(num_wires: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate quantum states and targets."""
    thetas = 2 * math.pi * torch.rand(samples)
    phis = 2 * math.pi * torch.rand(samples)
    omega0 = torch.zeros(2 ** num_wires, dtype=torch.complex64)
    omega0[0] = 1.0
    omega1 = torch.zeros(2 ** num_wires, dtype=torch.complex64)
    omega1[-1] = 1.0
    states = torch.stack([torch.cos(thetas[i]) * omega0 + torch.exp(1j * phis[i]) * torch.sin(thetas[i]) * omega1
                          for i in range(samples)], dim=0)
    labels = torch.sin(2 * thetas) * torch.cos(phis)
    return states, labels

class QuantumRegressionDataset(torch.utils.data.Dataset):
    """Dataset for quantum regression."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return self.states.size(0)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {"states": self.states[idx], "target": self.labels[idx]}

class QuantumRandomEncoder(tq.QuantumModule):
    """Random layer encoder inspired by QFCModel."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        return self.measure(qdev)

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)

class HybridRegression(tq.QuantumModule):
    """Quantum regression model: quantum encoder + classical transformer."""
    def __init__(
        self,
        num_wires: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = QuantumRandomEncoder(num_wires)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        qdev.set_state(state_batch)
        features = self.encoder(qdev)
        seq = features.unsqueeze(1)  # (bsz, 1, n_wires)
        seq = self.pos_encoder(seq)
        seq = self.transformers(seq)
        seq = seq.mean(dim=1)
        return self.head(seq).squeeze(-1)

__all__ = ["HybridRegression", "QuantumRegressionDataset", "generate_superposition_data"]
