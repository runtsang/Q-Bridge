"""Hybrid transformer with quantum circuits for attention, feed‑forward and patch embedding.

The module extends the original QTransformerTorch by:
- Adding a QuantumPatchEmbedding that applies a random quantum kernel to each token vector.
- Using quantum‑inspired multi‑head attention and feed‑forward blocks that employ random quantum layers.
- Clipping quantum gate parameters to the range [-5, 5] to mirror the fraud‑detection style regularisation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumPatchEmbedding(tq.QuantumModule):
    """Apply a random quantum kernel to each token vector."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.n_wires = embed_dim
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Clip gate parameters
        for gate in self.q_layer.parameters():
            if hasattr(gate, "params"):
                gate.params = gate.params.clamp(-5.0, 5.0)
            else:
                gate.param = gate.param.clamp(-5.0, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, embed_dim = x.shape
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz * seq_len, device=device)
        flat = x.reshape(bsz * seq_len, embed_dim)
        self.encoder(qdev, flat)
        self.q_layer(qdev)
        measurement = self.measure(qdev)
        return measurement.reshape(bsz, seq_len, embed_dim)


class QuantumMultiHeadAttention(tq.QuantumModule):
    """Quantum‑enabled multi‑head attention."""
    class QLayer(tq.QuantumModule):
        def __init__(self, d_k: int) -> None:
            super().__init__()
            self.n_wires = d_k
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
            )
            self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)
            # Clip parameters
            for gate in self.q_layer.parameters():
                if hasattr(gate, "params"):
                    gate.params = gate.params.clamp(-5.0, 5.0)
                else:
                    gate.param = gate.param.clamp(-5.0, 5.0)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            self.q_layer(q_device)
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(self.d_k)
        self.q_device = q_device
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        qdev = self.q_device or tq.QuantumDevice(self.q_layer.n_wires,
                                                 bsz=batch_size,
                                                 device=x.device)
        k = self.q_layer(x, qdev).view(batch_size, seq_len, self.num_heads, self.d_k)
        q = self.q_layer(x, qdev).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.q_layer(x, qdev).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine(out)


class QuantumFeedForward(tq.QuantumModule):
    """Feed‑forward block realised with a quantum layer."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
            )
            self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_qubits)))
            self.measure = tq.MeasureAll(tq.PauliZ)
            # Clip parameters
            for gate in self.q_layer.parameters():
                if hasattr(gate, "params"):
                    gate.params = gate.params.clamp(-5.0, 5.0)
                else:
                    gate.param = gate.param.clamp(-5.0, 5.0)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            self.q_layer(q_device)
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_qubits_ffn: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_qubits_ffn)
        self.q_device = q_device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qdev = self.q_device or tq.QuantumDevice(self.q_layer.n_wires,
                                                 bsz=batch_size,
                                                 device=x.device)
        flat = x.reshape(batch_size * seq_len, -1)
        self.q_layer(flat, qdev)
        quantum_out = self.q_layer.measure(qdev).reshape(batch_size, seq_len, self.q_layer.n_wires)
        out = self.linear1(quantum_out)
        out = self.linear2(self.dropout(F.relu(out)))
        return out


class TransformerBlock(tq.QuantumModule):
    """Hybrid transformer block combining quantum attention and feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 q_device: Optional[tq.QuantumDevice] = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = QuantumMultiHeadAttention(embed_dim, num_heads, dropout, q_device=q_device)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits_ffn, dropout, q_device=q_device)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class QTransformerGen320(tq.QuantumModule):
    """Quantum‑enhanced transformer classifier with optional quantum patch embedding."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 q_device: Optional[tq.QuantumDevice] = None,
                 quantum_patch: bool = False) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoder(embed_dim)
        self.quantum_patch = quantum_patch
        if quantum_patch:
            self.patch_embed = QuantumPatchEmbedding(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlock(embed_dim,
                               num_heads,
                               ffn_dim,
                               n_qubits_transformer,
                               n_qubits_ffn,
                               q_device=q_device,
                               dropout=dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        if self.quantum_patch:
            tokens = self.patch_embed(tokens)
        x = self.positional(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "QuantumPatchEmbedding",
    "QuantumMultiHeadAttention",
    "QuantumFeedForward",
    "TransformerBlock",
    "PositionalEncoder",
    "QTransformerGen320",
]
