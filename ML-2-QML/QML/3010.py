"""HybridQuanvolutionTransformer – quantum‑enhanced implementation.

This module re‑implements the same pipeline as the classical version but
replaces the convolutional filter and transformer blocks with quantum
modules built on TorchQuantum.  The quanvolution filter now maps each
2×2 patch through an 8‑qubit random circuit, while the transformer
blocks use a quantum attention head and a quantum feed‑forward network.
The API remains identical to the classical module, allowing seamless
switching between modes."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
# 1. Quantum quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to each 2×2 image patch."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)  # (B, 4*14*14)


# --------------------------------------------------------------------------- #
# 2. Positional encoding (identical to classical)
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding compatible with a single token."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# 3. Quantum attention head
# --------------------------------------------------------------------------- #
class QuantumAttention(tq.QuantumModule):
    """Multi‑head attention replaced by an 8‑wire random quantum circuit."""
    def __init__(self, n_wires: int = 8, device: str | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)), device=device)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (B, seq_len, embed_dim == n_wires)
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        qdev = tq.QuantumDevice(self.n_wires, bsz=B * L, device=x.device)
        self.encoder(qdev, x_flat)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return out.reshape(B, L, D)


# --------------------------------------------------------------------------- #
# 4. Quantum feed‑forward network
# --------------------------------------------------------------------------- #
class FeedForwardQuantum(tq.QuantumModule):
    """Feed‑forward network realized by a quantum circuit followed by classical linear layers."""
    def __init__(self, n_qubits: int, ffn_dim: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        qdev = tq.QuantumDevice(self.n_qubits, bsz=B * L, device=x.device)
        self.encoder(qdev, x_flat)
        for wire, gate in enumerate(self.parameters):
            gate(qdev, wires=wire)
        out = self.measure(qdev)
        out = out.reshape(B, L, D)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out


# --------------------------------------------------------------------------- #
# 5. Quantum transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses quantum attention and quantum feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = QuantumAttention(n_wires=embed_dim)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(attn_out)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# 6. Top‑level hybrid transformer with quantum filter
# --------------------------------------------------------------------------- #
class HybridQuanvolutionTransformerQuantum(nn.Module):
    """Image‑to‑sequence classifier that uses a quantum quanvolution filter
    followed by a stack of quantum transformer blocks."""
    def __init__(
        self,
        vocab_size: int = 0,  # unused but kept for API symmetry
        embed_dim: int = 8,
        num_heads: int = 1,
        num_blocks: int = 4,
        ffn_dim: int = 16,
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilterQuantum()
        self.token_embed = nn.Linear(4 * 14 * 14, embed_dim)
        self.pos_embed = PositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.filter(x)  # (B, 4*14*14)
        x = self.token_embed(x).unsqueeze(1)  # (B, 1, embed_dim)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)  # global average pooling over the single token
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "QuanvolutionFilterQuantum",
    "PositionalEncoding",
    "TransformerBlockQuantum",
    "HybridQuanvolutionTransformerQuantum",
]
