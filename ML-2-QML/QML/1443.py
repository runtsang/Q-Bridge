"""Quantum‑enhanced transformer modules using TorchQuantum.

Provides a parallel API to the classical implementation: the same classes
can be swapped in to a training pipeline without changing the high‑level
model definition.  Quantum sub‑modules are optional and default to the
classical variants if no quantum parameters are supplied.
"""

import math
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    """Base class for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑based attention that encodes each token with a variational circuit."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, input_vec: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, input_vec)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_wires: int = 8, qdevice: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires = n_wires
        self.q_layer = self._QLayer(n_wires)
        self.qdevice = qdevice or tq.QuantumDevice(n_wires=n_wires)
        self.input_proj = nn.Linear(embed_dim, n_wires, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        # Project token embeddings to qubit space
        token_vecs = self.input_proj(x)  # (batch, seq, n_wires)
        outputs = []
        for token in token_vecs.unbind(dim=1):
            qdev = self.qdevice.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        quantum_out = torch.stack(outputs, dim=1)  # (batch, seq, n_wires)
        return self.out_proj(quantum_out)


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a quantum circuit followed by classical layers."""
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, input_vec: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, input_vec)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.q_layer = self._QLayer(n_qubits)
        self.qdevice = tq.QuantumDevice(n_qubits)
        self.input_proj = nn.Linear(embed_dim, n_qubits, bias=False)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.qdevice.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)  # (batch, seq, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    """Transformer block that optionally uses quantum attention/FFN."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        attention_cls: Type[MultiHeadAttentionBase] = MultiHeadAttentionQuantum,
        ffn_cls: Type[FeedForwardBase] = FeedForwardQuantum,
        dropout: float = 0.1,
        gate_residual: bool = False,
    ):
        super().__init__()
        self.attn = attention_cls(embed_dim, num_heads, dropout)
        self.ffn = ffn_cls(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate_residual = gate_residual

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        if self.gate_residual:
            x = x + self.dropout(attn_out) * torch.sigmoid(self.dropout(attn_out))
        else:
            x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        return self.norm2(x)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding (identical to the classical version)."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) *
            (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class QuantumTextClassifier(nn.Module):
    """Transformer‑based text classifier that can use quantum sub‑modules."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        attention_cls: Type[MultiHeadAttentionBase] = MultiHeadAttentionQuantum,
        ffn_cls: Type[FeedForwardBase] = FeedForwardQuantum,
        gate_residual: bool = False,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    attention_cls=attention_cls,
                    ffn_cls=ffn_cls,
                    dropout=dropout,
                    gate_residual=gate_residual,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "QuantumTextClassifier",
    "PositionalEncoder",
]
