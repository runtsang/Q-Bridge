"""Quantum‑enhanced transformer with variational circuits and a learnable attention gate.

The implementation builds on the hybrid design from the classical module but replaces the
simulated quantum parts with real quantum circuits using TorchQuantum.  A learnable
gate controls the trade‑off between classical dot‑product attention and a variational
circuit per head.  The feed‑forward stage similarly uses a variational circuit that
can be reset for reproducible training.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class MultiHeadAttentionBase(nn.Module):
    """Base class for attention modules."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

class HybridAttention(MultiHeadAttentionBase):
    """Quantum‑classical attention with a learnable gate."""
    class QLayer(tq.QuantumModule):
        """Variational circuit for a single head."""
        def __init__(self, d_k: int, n_layers: int = 2):
            super().__init__()
            self.n_wires = d_k
            self.n_layers = n_layers
            # Encoder maps each input dimension to a wire
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(d_k)
                ]
            )
            # Trainable rotations
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(d_k)]
            )
            # Entanglement
            self.cnot_pattern = [(i, i + 1) for i in range(d_k - 1)] + [(d_k - 1, 0)]
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for gate in self.parameters:
                gate(q_device, wires=gate.wires)
            for src, tgt in self.cnot_pattern:
                tqf.cnot(q_device, wires=[src, tgt])
            return self.measure(q_device)

        def reset_parameters(self) -> None:
            for gate in self.parameters:
                nn.init.uniform_(gate.weight, -math.pi, math.pi)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_layers: int = 2) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.classical_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.d_k = embed_dim // num_heads
        self.q_layers = nn.ModuleList(
            [self.QLayer(self.d_k, n_layers) for _ in range(num_heads)]
        )
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Classical attention
        attn_out, _ = self.classical_attn(x, x, x, key_padding_mask=mask)
        # Quantum attention
        batch, seq_len, _ = x.size()
        x_reshaped = x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq, d_k)
        q_heads = []
        for i in range(self.num_heads):
            head = x_reshaped[:, i, :, :]  # (batch, seq, d_k)
            head_flat = head.contiguous().view(batch * seq_len, self.d_k)
            qdev = tq.QuantumDevice(n_wires=self.d_k, bsz=batch * seq_len, device=head_flat.device)
            out = self.q_layers[i](head_flat, qdev)
            out = out.view(batch, seq_len, self.d_k)
            q_heads.append(out)
        q_heads = torch.stack(q_heads, dim=1)  # (batch, heads, seq, d_k)
        q_heads = q_heads.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        gate = torch.sigmoid(self.gate)
        out = (1 - gate) * attn_out + gate * q_heads
        return self.dropout(out)

    def reset_parameters(self) -> None:
        for ql in self.q_layers:
            ql.reset_parameters()

class FeedForwardBase(nn.Module):
    """Base feed‑forward layer."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardQuantum(FeedForwardBase):
    """Variational quantum feed‑forward network."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int, n_layers: int = 2):
            super().__init__()
            self.n_wires = n_qubits
            self.n_layers = n_layers
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_qubits)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.cnot_pattern = [(i, i + 1) for i in range(n_qubits - 1)] + [(n_qubits - 1, 0)]
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for gate in self.parameters:
                gate(q_device, wires=gate.wires)
            for src, tgt in self.cnot_pattern:
                tqf.cnot(q_device, wires=[src, tgt])
            return self.measure(q_device)

        def reset_parameters(self) -> None:
            for gate in self.parameters:
                nn.init.uniform_(gate.weight, -math.pi, math.pi)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1, n_layers: int = 2) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits, n_layers)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # (batch, seq, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

    def reset_parameters(self) -> None:
        self.q_layer.reset_parameters()

class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        n_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = HybridAttention(embed_dim, num_heads, dropout, n_layers)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout, n_layers)
        else:
            self.ffn = FeedForwardBase(embed_dim, ffn_dim, dropout)

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
        return x + self.pe[:, : x.size(1)]

class TextClassifier(nn.Module):
    """Transformer‑based text classifier with optional quantum submodules."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        blocks = [
            TransformerBlockQuantum(
                embed_dim,
                num_heads,
                ffn_dim,
                n_qubits_transformer,
                n_qubits_ffn,
                n_layers,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "HybridAttention",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifier",
]
