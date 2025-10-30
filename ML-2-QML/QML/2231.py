"""Hybrid transformer classifier – quantum implementation.

The quantum version replaces the classical attention and feed‑forward layers
with variational circuits implemented via TorchQuantum.  A convolutional
feature extractor is also provided in a quantum form, mirroring the
classical counterpart.  The API is identical to the classical module,
allowing side‑by‑side experiments with the same hyper‑parameters.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Optional, Literal

# --------------------------------------------------------------------------- #
# Quantum sub‑modules
# --------------------------------------------------------------------------- #

class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Multi‑head attention where each head is a small variational circuit."""

    class QHead(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([
                tq.RX(has_params=True, trainable=True) for _ in range(n_wires)
            ])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.parameters, range(self.n_wires)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.head_dim)
        self.heads = nn.ModuleList([self.QHead(self.head_dim) for _ in range(num_heads)])

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply quantum head to each head tensor
        q_q = torch.stack([self.heads[i](q[:, i, :, :], self.q_device) for i in range(self.num_heads)], dim=1)
        k_q = torch.stack([self.heads[i](k[:, i, :, :], self.q_device) for i in range(self.num_heads)], dim=1)
        v_q = torch.stack([self.heads[i](v[:, i, :, :], self.q_device) for i in range(self.num_heads)], dim=1)

        attn_scores = torch.matmul(q_q, k_q.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v_q)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)


class FeedForwardQuantum(tq.QuantumModule):
    """Feed‑forward network realised by a quantum circuit followed by classical layers."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, ffn_dim: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([
                tq.RY(has_params=True, trainable=True) for _ in range(n_wires)
            ])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.parameters, range(self.n_wires)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = self.QLayer(n_qubits, ffn_dim)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        out = torch.zeros(B, T, self.q_layer.n_wires, device=x.device)
        for i in range(T):
            out[:, i, :] = self.q_layer(x[:, i, :], self.q_device)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(tq.QuantumModule):
    """Transformer block that can mix classical and quantum sub‑modules."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        use_quantum_attention: bool = True,
        use_quantum_ffn: bool = True,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        if use_quantum_attention:
            self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)

        if use_quantum_ffn:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class PositionalEncoder(tq.QuantumModule):
    """Same sinusoidal encoder as the classical version."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# --------------------------------------------------------------------------- #
# Quantum convolutional front‑end
# --------------------------------------------------------------------------- #

class QuantumFeatureExtractor(tq.QuantumModule):
    """Quantum‑enhanced convolutional encoder inspired by QuantumNAT."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3)
            tqf.sx(qdev, wires=2)
            tqf.cnot(qdev, wires=[3, 0])

    def __init__(self, output_dim: int):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.fc = nn.Linear(self.n_wires, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=B, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(B, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        return self.fc(out)


# --------------------------------------------------------------------------- #
# Public hybrid classifier – quantum variant
# --------------------------------------------------------------------------- #

class HybridTransformerClassifier(tq.QuantumModule):
    """
    Quantum‑enhanced transformer classifier mirroring the classical API.

    Parameters
    ----------
    front_end : Literal["text", "image"]
        Choose between a token embedding front‑end or a quantum convolutional encoder.
    vocab_size : int, optional
        Vocabulary size for text input. Ignored for image front‑end.
    embed_dim : int
        Dimensionality of the token / feature representation.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of the feed‑forward network.
    num_classes : int
        Number of target classes.
    dropout : float
        Drop‑out probability.
    use_quantum_attention : bool
        Whether to use quantum attention in each block.
    use_quantum_ffn : bool
        Whether to use quantum feed‑forward in each block.
    n_qubits_transformer : int
        Number of qubits per transformer attention head.
    n_qubits_ffn : int
        Number of qubits per feed‑forward quantum layer.
    """
    def __init__(
        self,
        front_end: Literal["text", "image"],
        vocab_size: Optional[int] = None,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_quantum_attention: bool = True,
        use_quantum_ffn: bool = True,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 4,
    ):
        super().__init__()
        self.front_end_type = front_end

        if front_end == "text":
            if vocab_size is None:
                raise ValueError("vocab_size must be provided for text front‑end")
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoder = PositionalEncoder(embed_dim)
        else:
            self.front_end = QuantumFeatureExtractor(output_dim=embed_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlockQuantum(
                embed_dim,
                num_heads,
                ffn_dim,
                use_quantum_attention=use_quantum_attention,
                use_quantum_ffn=use_quantum_ffn,
                n_qubits_transformer=n_qubits_transformer,
                n_qubits_ffn=n_qubits_ffn,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.front_end_type == "text":
            x = self.token_embedding(x)
            x = self.pos_encoder(x)
        else:
            x = self.front_end(x)

        for block in self.transformer_blocks:
            x = block(x)

        if self.front_end_type == "text":
            x = x.mean(dim=1)

        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumFeatureExtractor",
    "HybridTransformerClassifier",
]
