"""Quantum‑enhanced hybrid quanvolution‑transformer.

The QML version keeps the same public API but replaces the filter with a quantum
kernel and the transformer with quantum‑aware sub‑modules.  The module is
designed to be dropped in at the same path as the classical variant, allowing
experiments that swap only the quantum parts while leaving the rest of the
training pipeline untouched.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# Quantum quanvolution filter
class HybridQuanvolutionFilter(tq.QuantumModule):
    """Quantum kernel that processes 2×2 image patches."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                patches.append(self.measure(qdev).view(bsz, 4))
        return torch.cat(patches, dim=1)

# Classical transformer block used as fallback
class TransformerBlock(nn.Module):
    """Classical transformer block used as fallback in quantum module."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# Quantum‑aware transformer blocks
class QMultiHeadAttention(tq.QuantumModule):
    """Multi‑head attention where each head is a small quantum circuit."""
    class _QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)
                ]
            )
            self.param_gates = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.param_gates:
                gate(qdev)
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self._QLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        x = x.transpose(2, 1).contiguous()  # (B, num_heads, seq_len, d_k)
        outputs = []
        for head in range(self.num_heads):
            qdev = tq.QuantumDevice(self.q_layer.n_wires, bsz=batch_size, device=x.device)
            head_out = self.q_layer(x[:, head], qdev)
            outputs.append(head_out)
        out = torch.stack(outputs, dim=1)  # (B, num_heads, 4)
        out = out.view(batch_size, seq_len, self.embed_dim)
        return out

class QFeedForward(tq.QuantumModule):
    """Feed‑forward network realised by a quantum module."""
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.param_gates = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.param_gates:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits
        self.q_layer = self._QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        seq_len = x.size(1)
        outputs = []
        for i in range(seq_len):
            qdev = tq.QuantumDevice(self.n_qubits, bsz=batch_size, device=x.device)
            out = self.q_layer(x[:, i], qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(out)
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(tq.QuantumModule):
    """Quantum‑aware transformer block."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QMultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = QFeedForward(embed_dim, ffn_dim, n_qubits=8)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(tq.QuantumModule):
    """Sinusoidal positional encoding wrapped for quantum modules."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]

class HybridQuanvolutionClassifier(tq.QuantumModule):
    """End‑to‑end quantum‑enhanced classifier."""
    def __init__(
        self,
        num_classes: int,
        *,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        quantum: bool = False,
    ) -> None:
        super().__init__()
        self.filter = HybridQuanvolutionFilter()
        self.pos_enc = PositionalEncoder(embed_dim)
        self.token_proj = nn.Linear(4, embed_dim)
        if quantum:
            self.transformer = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformer = nn.Sequential(
                *[
                    TransformerBlock(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats = self.filter(x)                           # (B, 4*14*14)
        seq = feats.view(x.size(0), -1, 4)                # (B, 196, 4)
        seq = self.token_proj(seq)                        # (B, 196, embed_dim)
        seq = self.pos_enc(seq)                           # (B, 196, embed_dim)
        seq = self.transformer(seq)                       # (B, 196, embed_dim)
        out = seq.mean(dim=1)                             # (B, embed_dim)
        out = self.dropout(out)
        return self.classifier(out)

__all__ = [
    "HybridQuanvolutionFilter",
    "HybridQuanvolutionClassifier",
]
