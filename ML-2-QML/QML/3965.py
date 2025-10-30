"""Quantum‑centric modules for the integrated architecture.

Provides a quantum feed‑forward network that can be plugged into the
Transformer blocks of ``IntegratedQuanvolutionTransformer`` when
quantum capabilities are desired.  The implementation uses TorchQuantum
and expects the library to be installed.  If TorchQuantum is not
available, importing this module will raise an ImportError.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf

# Quantum feed‑forward layer
class FeedForwardQuantum(nn.Module):
    """Feed‑forward network implemented via a small quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        if n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)

        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        batch, seq_len, _ = x.size()
        outputs = []
        for token in x.unbind(dim=1):
            qdev_token = tq.QuantumDevice(self.n_qubits, bsz=token.size(0), device=token.device)
            self.encoder(qdev_token, token)
            for wire, gate in enumerate(self.params):
                gate(qdev_token, wires=wire)
            out = self.measure(qdev_token)  # (batch, n_qubits)
            out = self.linear1(self.dropout(out))
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # (batch, seq_len, ffn_dim)
        return self.linear2(F.relu(out))

# Transformer block that mixes classical attention with the quantum feed‑forward
class TransformerBlockQuantum(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_ffn: int,
                 dropout: float = 0.1):
        super().__init__()
        if n_qubits_ffn <= 0:
            raise ValueError("n_qubits_ffn must be positive for a quantum block")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # Classical multi‑head attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # Quantum feed‑forward
        ffn_out = self.ffn(x, q_device)
        return self.norm2(x + self.dropout(ffn_out))

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

# Quantum‑enhanced integrated model
class IntegratedQuanvolutionTransformer(nn.Module):
    """Full model that applies a quanvolution filter to an image, embeds the resulting patches,
    passes them through a transformer that uses quantum feed‑forward sub‑modules,
    and produces a classification score."""
    def __init__(self,
                 in_channels: int = 1,
                 depth_channels: int = 4,
                 embed_dim: int = 64,
                 num_heads: int = 8,
                 num_blocks: int = 4,
                 ffn_dim: int = 256,
                 n_qubits_ffn: int = 8,
                 num_classes: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        # Classical quanvolution front‑end
        self.qfilter = nn.Conv2d(in_channels, depth_channels, kernel_size=2,
                                 stride=2, groups=in_channels)
        self.pointwise = nn.Conv2d(depth_channels, in_channels, kernel_size=1)
        self.embed_proj = nn.Linear(depth_channels, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        # Quantum transformer blocks
        self.transformer = nn.ModuleList(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                      n_qubits_ffn, dropout)
              for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.dropout = nn.Dropout(dropout)
        self.n_qubits_ffn = n_qubits_ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # 1. Classical quanvolution
        feat = self.qfilter(x)                     # (B, depth_channels, H/2, W/2)
        feat = self.pointwise(feat)                # (B, C, H/2, W/2)
        seq_len = 14 * 14
        feat = feat.view(x.size(0), seq_len, -1)   # (B, seq_len, depth_channels)
        # 2. Project to embedding dimension
        feat = self.embed_proj(feat)               # (B, seq_len, embed_dim)
        # 3. Positional encoding
        feat = self.pos_encoder(feat)              # (B, seq_len, embed_dim)
        # 4. Quantum transformer blocks
        q_device = tq.QuantumDevice(n_wires=self.n_qubits_ffn, bsz=x.size(0), device=x.device)
        out = feat
        for block in self.transformer:
            out = block(out, q_device)
        # 5. Pooling and classification
        out = out.mean(dim=1)                      # (B, embed_dim)
        out = self.dropout(out)
        return self.classifier(out)

__all__ = ["FeedForwardQuantum", "TransformerBlockQuantum", "IntegratedQuanvolutionTransformer"]
