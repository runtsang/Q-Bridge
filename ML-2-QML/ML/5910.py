from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return features and targets for the regression task.
    The data are sampled from a sinusoidal superposition that matches the
    original quantum distribution, but the implementation stays fully
    classical.  Using tensors keeps the API compatible with the quantum
    model which expects a ``torch.Tensor`` input.
    """
    x = torch.randn(samples, num_features, dtype=torch.float32)
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x, y.float()

class RegressionDataset(Dataset):
    """Dataset that yields ``states`` and ``target`` tensors for training."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {"states": self.features[idx], "target": self.labels[idx]}

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding compatible with the original transformer."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                        (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """Standard multi‑head attention – kept from the ML seed."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = q.size()
        qkv = self.qkv(q).view(batch, seq, 3, self.head_dim, self.num_heads)
        qkv = qkv.permute(2, 0, 1, 3, 4)   # (3, B, seq, h, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v).transpose(1, 2).contiguous()
        return self.out(out.reshape(batch, seq, -1))

class FeedForward(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer block that can optionally host a quantum sub‑module."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 use_quantum: bool = False, q_module: nn.Module | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.use_quantum = use_quantum
        self.q_module = q_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        if self.use_quantum and self.q_module is not None:
            x = self.q_module(x)
        else:
            x = self.ffn(x)
        return self.norm2(x)

class TextClassifier(nn.Module):
    """Classifier that merges the classical transformer with a quantum head."""
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_blocks: int, ffn_dim: int, num_classes: int,
                 dropout: float = 0.1, use_quantum: bool = False,
                 q_head: nn.Module | None = None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, use_quantum, q_head)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_enc(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.head(x)

class QuantumEncoder:
    """Quantum encoder that maps a classical vector into a quantum state."""
    def __init__(self, num_wires: int):
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, x)
        self.q_layer(qdev)
        return self.measure(qdev)

class UnifiedQModel(nn.Module):
    """Hybrid model that stacks a transformer backbone with a quantum encoder."""
    def __init__(self, num_features: int, num_wires: int,
                 embed_dim: int = 32, num_heads: int = 4,
                 ffn_dim: int = 64, num_blocks: int = 2,
                 use_quantum_head: bool = True):
        super().__init__()
        # Classical transformer for context
        self.transformer = TextClassifier(
            vocab_size=num_features,  # treat each feature as a token
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ffn_dim=ffn_dim,
            num_classes=1,
            dropout=0.1,
            use_quantum=False,
        )
        # Quantum encoder that mirrors the original QML seed
        self.quantum_encoder = QuantumEncoder(num_wires=num_wires)

        # Final linear head – maps quantum output to a regression target
        self.head = nn.Linear(num_wires, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        1. Pass the raw feature vectors through the transformer to obtain
           a contextual embedding.
        2. Feed the embedding into the quantum circuit to encode
           *all* the features in one shot.
        3. Measure and project the expectation values to a single output.
        """
        # Classical part
        ctx = self.transformer(states)
        # Quantum part – one circuit per batch element
        batch_size = states.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.quantum_encoder.n_wires, bsz=batch_size, device=states.device)
        quantum_state = self.quantum_encoder(qdev, ctx)
        # Expectation values – we return a tensor of size (bs, n_wires)
        return self.head(quantum_state).squeeze(-1)

__all__ = [
    "QModel",
    "RegressionDataset",
    "generate_superposition_data",
]
