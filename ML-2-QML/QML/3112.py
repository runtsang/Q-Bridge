"""
Hybrid transformer text classifier – quantum implementation.
The quantum branch replaces the convolutional front‑end with a
parameter‑aware quantum encoder and optionally swaps the transformer
blocks for their quantum counterparts.  All quantum circuits are
defined with TorchQuantum and can be executed on simulators or
back‑ends such as AWS Braket.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# ────────────────────────────────────────────────────────────────────────
#  Quantum front‑end
# ────────────────────────────────────────────────────────────────────────
class QuantumFrontEnd(tq.QuantumModule):
    """
    Quantum encoder that maps a 4‑dimensional classical feature vector
    to a 4‑qubit state, applies a depth‑2 random layer, and measures
    all qubits.  The output is a 4‑dimensional tensor suitable for
    downstream layers.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # encode classical 4‑dim vector
        self.encoder(qdev, x)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


# ────────────────────────────────────────────────────────────────────────
#  Quantum transformer primitives
# ────────────────────────────────────────────────────────────────────────
class QuantumMultiHeadAttention(tq.QuantumModule):
    """Attention that projects queries, keys and values through quantum circuits."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
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
        self.n_wires = n_wires
        self.q_layer = self.QLayer(n_wires)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def _apply_q(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        out = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=token.size(0), device=token.device)
            out.append(self.q_layer(token, qdev))
        return torch.stack(out, dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self._apply_q(x)
        k = self._apply_q(x)
        v = self._apply_q(x)
        # classical attention on the quantum‑encoded representations
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return self.combine(out)


class QuantumFeedForward(tq.QuantumModule):
    """Feed‑forward implemented with a quantum layer followed by classical MLP."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "ry", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int, dropout: float = 0.1):
        super().__init__()
        self.n_wires = n_wires
        self.q_layer = self.QLayer(n_wires)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        out = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=token.size(0), device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class QuantumTransformerBlock(tq.QuantumModule):
    """Single transformer block where both attention and feed‑forward are quantum."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_wires_attn: int, n_wires_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumMultiHeadAttention(embed_dim, num_heads, dropout, n_wires_attn)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_wires_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


# ────────────────────────────────────────────────────────────────────────
#  Positional encoder (identical to classical)
# ────────────────────────────────────────────────────────────────────────
class PositionalEncoder(tq.QuantumModule):
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


# ────────────────────────────────────────────────────────────────────────
#  Hybrid classifier – quantum back‑end
# ────────────────────────────────────────────────────────────────────────
class HybridTextClassifier(tq.QuantumModule):
    """
    Quantum‑enabled counterpart of the classical classifier.
    All interfaces mirror the classical version; the only difference
    is that the front‑end and optionally the transformer blocks are
    instantiated with quantum modules.
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
        n_wires_attn: int = 8,
        n_wires_ffn: int = 8,
        use_quantum_front_end: bool = True,
    ) -> None:
        super().__init__()
        self.front_end = QuantumFrontEnd() if use_quantum_front_end else None
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[
                QuantumTransformerBlock(
                    embed_dim, num_heads, ffn_dim, n_wires_attn, n_wires_ffn, dropout
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # image → quantum feature vector
        if self.front_end is not None and x.dim() == 4:
            x = self.front_end(x)          # (batch, 4)
            x = x.unsqueeze(1)             # (batch, 1, 4)
            x = self.token_embedding(x)    # embed_dim channel
        else:
            x = self.token_embedding(x)    # (batch, seq_len, embed_dim)

        x = self.pos_embedding(x)
        x = self.transformers(x)
        x = x.mean(dim=1)  # global pooling
        x = self.dropout(x)
        return self.classifier(x)


__all__ = ["HybridTextClassifier"]
