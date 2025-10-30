"""
Quantum‑enhanced modules for UnifiedClassifierHybrid.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# ---------------------------------------------
# Quantum multi‑head attention
# ---------------------------------------------
class MultiHeadAttentionQuantum(nn.Module):
    """
    Attention where each projection is processed through a small variational circuit.
    """
    class _QHead(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, gate in enumerate(self.params):
                gate(q_device, wires=w)
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 n_wires_per_head: int = 4,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_heads = nn.ModuleList([self._QHead(n_wires_per_head) for _ in range(num_heads)])
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_wires_per_head)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        proj = self.proj(x)
        batch, seq, _ = proj.shape
        outputs = []
        for head, q_head in zip(proj.unbind(dim=2), self.q_heads):
            out = []
            for token in head.unbind(dim=0):
                qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
                out.append(q_head(token, qdev))
            outputs.append(torch.stack(out, dim=0))
        quantum_out = torch.stack(outputs, dim=2).reshape(batch, seq, self.embed_dim)
        return self.combine(quantum_out)

# ---------------------------------------------
# Quantum feed‑forward
# ---------------------------------------------
class FeedForwardQuantum(nn.Module):
    """
    Two‑layer feed‑forward with a quantum circuit between layers.
    """
    class _QFF(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, gate in enumerate(self.params):
                gate(q_device, wires=w)
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_wires: int,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self._QFF(n_wires)
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantum_out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            quantum_out.append(self.q_layer(token, qdev))
        quantum_out = torch.stack(quantum_out, dim=1)
        quantum_out = self.linear1(self.dropout(quantum_out))
        return self.linear2(F.relu(quantum_out))

# ---------------------------------------------
# Quantum transformer block
# ---------------------------------------------
class TransformerBlockQuantum(nn.Module):
    """
    Transformer block that substitutes both attention and feed‑forward
    with their quantum counterparts.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_wires_per_head: int,
                 n_wires_ffn: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim,
                                              num_heads,
                                              dropout,
                                              n_wires_per_head)
        self.ffn = FeedForwardQuantum(embed_dim,
                                      ffn_dim,
                                      n_wires_ffn,
                                      dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ---------------------------------------------
# Positional encoding (identical to classical)
# ---------------------------------------------
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# ---------------------------------------------
# Quantum text classifier
# ---------------------------------------------
class TextClassifierQuantum(nn.Module):
    """
    Transformer‑based classifier that can be instantiated entirely with quantum sub‑modules.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_wires_per_head: int = 4,
                 n_wires_ffn: int = 4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(*[
            TransformerBlockQuantum(embed_dim,
                                    num_heads,
                                    ffn_dim,
                                    n_wires_per_head,
                                    n_wires_ffn,
                                    dropout)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoding(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

# ---------------------------------------------
# Quantum classifier circuit (data‑uploading)
# ---------------------------------------------
def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a variational circuit that mirrors the classical build_classifier_circuit.
    Returns the circuit, an encoding ParameterVector, a weight ParameterVector,
    and a list of PauliZ observables.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    # Data encoding
    for q, param in zip(range(num_qubits), encoding):
        qc.rx(param, q)

    idx = 0
    for _ in range(depth):
        # Variational layer
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        # Entangling layer
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
        qc.cz(num_qubits - 1, 0)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]

    return qc, list(encoding), list(weights), observables

# ---------------------------------------------
# Helper: build_transformer_circuit (optional)
# ---------------------------------------------
def build_transformer_circuit(num_blocks: int,
                              embed_dim: int,
                              num_heads: int,
                              ffn_dim: int,
                              n_wires_per_head: int,
                              n_wires_ffn: int,
                              depth: int) -> tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Assemble a stack of quantum‑enhanced transformer blocks into a single circuit.
    Each block contributes a variational layer and an entangling layer.
    """
    qc = QuantumCircuit(embed_dim)
    encoding = ParameterVector("x", embed_dim)
    weights = ParameterVector("theta", embed_dim * depth * num_blocks * 2)

    # Data encoding
    for i, param in enumerate(encoding):
        qc.rx(param, i)

    idx = 0
    for _ in range(num_blocks):
        # Attention‑like variational layer
        for i in range(embed_dim):
            qc.ry(weights[idx], i)
            idx += 1
        # Entangling layer (CZ chain)
        for i in range(embed_dim - 1):
            qc.cz(i, i + 1)
        qc.cz(embed_dim - 1, 0)

        # Feed‑forward variational layer
        for i in range(embed_dim):
            qc.ry(weights[idx], i)
            idx += 1
        for i in range(embed_dim - 1):
            qc.cz(i, i + 1)
        qc.cz(embed_dim - 1, 0)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (embed_dim - i - 1))
                   for i in range(embed_dim)]

    return qc, list(encoding), list(weights), observables

__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifierQuantum",
    "build_classifier_circuit",
    "build_transformer_circuit",
]
