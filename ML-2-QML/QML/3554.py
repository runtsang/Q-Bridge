"""
Quantum implementation of the UnifiedRegressionTransformer.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumAttention(nn.Module):
    """Quantum multi‑head attention that encodes projections on a quantum device."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.parameters, range(self.n_wires)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 8):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.qlayer = self.QLayer(n_wires=self.d_k)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        x_head = x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, T, Dk)
        head_outs = []
        for h in range(self.num_heads):
            qdev = tq.QuantumDevice(n_wires=self.qlayer.n_wires, bsz=B, device=x.device)
            flat = x_head[:, h, :, :].reshape(-1, self.d_k)
            out_q = self.qlayer(flat, qdev)
            head_outs.append(out_q.view(B, T, self.d_k))
        proj = torch.stack(head_outs, dim=1)  # (B, H, T, Dk)
        scores = torch.matmul(proj, proj.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        attn = torch.matmul(scores, proj)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.combine(attn)

class QuantumFeedForward(nn.Module):
    """Quantum feed‑forward network."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.parameters, range(self.n_qubits)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.qlayer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.qlayer.n_qubits, bsz=token.size(0), device=token.device)
            out_q = self.qlayer(token, qdev)
            outputs.append(out_q)
        out = torch.stack(outputs, dim=1)  # (B, T, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class QuantumTransformerBlock(nn.Module):
    """Transformer block that can operate in quantum mode."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_attn: int = 8,
        n_qubits_ffn: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = QuantumAttention(embed_dim, num_heads, dropout, n_wires=n_qubits_attn)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
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
        return x + self.pe[:, :x.size(1)]

class UnifiedRegressionTransformer(nn.Module):
    """
    Quantum‑backed regression transformer.
    Mirrors the classical API but uses quantum modules internally.
    """
    def __init__(
        self,
        num_features: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        hidden_size: int | None = None,
        quantum_config: dict | None = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size or embed_dim

        # Feature encoder
        self.feature_proj = nn.Linear(num_features, embed_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoder(embed_dim)

        # Build transformer blocks
        blocks = [
            QuantumTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                n_qubits_attn=quantum_config.get("n_qubits_attn", 8) if quantum_config else 8,
                n_qubits_ffn=quantum_config.get("n_qubits_ffn", 8) if quantum_config else 8,
                dropout=quantum_config.get("dropout", 0.1) if quantum_config else 0.1,
            )
            for _ in range(num_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)

        # Output head
        self.head = nn.Linear(self.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)

__all__ = [
    "UnifiedRegressionTransformer",
    "QuantumTransformerBlock",
    "QuantumAttention",
    "QuantumFeedForward",
    "PositionalEncoder",
]
