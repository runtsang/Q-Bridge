"""Quantum‑enhanced hybrid model: CNN → quantum projection → quantum transformer classifier."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# ------------------- CNN Feature Extractor ------------------- #
class ResCNN(nn.Module):
    """Classical CNN with residual connection."""
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.res = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.norm = nn.BatchNorm2d(base_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return self.norm(x2 + self.res(x2))

# ------------------- Quantum Projection ------------------- #
class QuantumProjection(tq.QuantumModule):
    """Quantum projection of flattened CNN features to embedding dimension."""
    def __init__(self, in_features: int, embed_dim: int, n_wires: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(min(n_wires, in_features))
            ]
        )
        self.gates = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(n_wires, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=x.device, record_op=True)
        self.encoder(qdev, x)
        for w, gate in enumerate(self.gates):
            gate(qdev, wires=w)
        out = self.measure(qdev)
        return self.linear(out)

# ------------------- Positional Encoding ------------------- #
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

# ------------------- Quantum Attention ------------------- #
class QuantumAttention(tq.QuantumModule):
    """Multi‑head attention with quantum heads."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # quantum head per head
        self.q_heads = nn.ModuleList([QuantumProjection(self.head_dim, self.head_dim, n_wires) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        # quantum processing on each head
        q_out = []
        for h in range(self.num_heads):
            head_out = self.q_heads[h](out[:, h, :, :])  # (B, seq, head_dim)
            q_out.append(head_out.unsqueeze(1))
        out = torch.cat(q_out, dim=1).view(batch, seq, -1)
        return self.out_proj(out)

# ------------------- Quantum Feed‑Forward ------------------- #
class QuantumFeedForward(tq.QuantumModule):
    """Feed‑forward with quantum transformation."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, n_wires: int = 8):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.q_layer = QuantumProjection(ffn_dim, ffn_dim, n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = F.relu(out)
        out = self.q_layer(out)
        return self.linear2(self.dropout(out))

# ------------------- Quantum Transformer Block ------------------- #
class QuantumTransformerBlock(tq.QuantumModule):
    """Single transformer block with quantum attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, n_wires: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, dropout, n_wires)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, dropout, n_wires)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)

# ------------------- Quantum Text Classifier ------------------- #
class QuantumTextClassifier(tq.QuantumModule):
    """Transformer‑based classifier with quantum sub‑modules."""
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_blocks: int, ffn_dim: int, num_classes: int,
                 dropout: float = 0.1, n_wires: int = 8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [QuantumTransformerBlock(embed_dim, num_heads, ffn_dim, dropout, n_wires)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

# ------------------- Hybrid Model ------------------- #
class QuantumNATHybrid(tq.QuantumModule):
    """Hybrid quantum‑classical model: CNN → quantum projection → quantum transformer."""
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 16,
                 vocab_size: int = 30522,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 num_blocks: int = 4,
                 ffn_dim: int = 128,
                 num_classes: int = 4,
                 dropout: float = 0.1,
                 n_wires: int = 8):
        super().__init__()
        self.cnn = ResCNN(in_channels, base_channels)
        self.proj = QuantumProjection(base_channels * 2 * 7 * 7, embed_dim, n_wires)
        self.classifier = QuantumTextClassifier(vocab_size, embed_dim, num_heads,
                                                num_blocks, ffn_dim, num_classes,
                                                dropout, n_wires)

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(img)
        feat = feat.view(feat.size(0), -1)
        feat = self.proj(feat)
        img_token = feat.unsqueeze(1)
        txt_token = self.classifier.token_emb(txt)
        tokens = torch.cat([img_token, txt_token], dim=1)
        tokens = self.classifier.pos_emb(tokens)
        x = tokens
        for block in self.classifier.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.classifier.dropout(x)
        return self.classifier.classifier(x)

__all__ = ["QuantumNATHybrid"]
