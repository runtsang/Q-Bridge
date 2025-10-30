"""Quantum‑enhanced transformer implementation.

This module mirrors the classical API but replaces the transformer
blocks with quantum‑aware variants built on top of TorchQuantum and
Qiskit.  The quantum modules are intentionally lightweight to keep
the example readable while still demonstrating how classical
operations can be wrapped with quantum circuits.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator

# --------------------------------------------------------------------------- #
# 1. Quantum building blocks
# --------------------------------------------------------------------------- #

class QLayer(tq.QuantumModule):
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        return self.measure(q_device)

class MultiHeadAttentionQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.q_layer = QLayer(self.d_k)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        batch, seq_len, dim = attn_output.shape
        flat = attn_output.view(-1, dim)
        qdev = self.q_device.copy(bsz=flat.size(0), device=flat.device)
        quantum_out = self.q_layer(flat[:, : self.d_k], qdev)
        out = torch.cat([quantum_out, flat[:, self.d_k:]], dim=1)
        out = out.view(batch, seq_len, dim)
        out = self.combine_heads(out)
        return self.dropout(out)

class FeedForwardQuantum(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, n_qubits: int = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.n_qubits = n_qubits or embed_dim
        self.q_layer = QLayer(self.n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        batch, seq_len, dim = x.shape
        flat = x.view(-1, dim)
        qdev = self.q_device.copy(bsz=flat.size(0), device=flat.device)
        quantum_out = self.q_layer(flat, qdev)
        x = quantum_out.view(batch, seq_len, dim)
        x = self.linear2(x)
        return self.dropout(x)

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
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

# --------------------------------------------------------------------------- #
# 2. Quantum classifier head
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

def EstimatorQNN_wrapper(num_qubits: int = 4, depth: int = 2) -> EstimatorQNN:
    qc, enc, wts, obs = build_classifier_circuit(num_qubits, depth)
    estimator = Estimator()
    return EstimatorQNN(
        circuit=qc,
        observables=obs,
        input_params=enc,
        weight_params=wts,
        estimator=estimator,
    )

# --------------------------------------------------------------------------- #
# 3. Hybrid text classifier (quantum‑aware)
# --------------------------------------------------------------------------- #

class HybridTextClassifier(nn.Module):
    """Transformer‑based text classifier that can mix classical and quantum sub‑modules.

    Parameters
    ----------
    vocab_size : int
        Size of the input vocabulary.
    embed_dim : int
        Dimension of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Feed‑forward hidden dimension.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Drop‑out probability.
    use_quantum_blocks : Optional[Iterable[bool]], optional
        Sequence of booleans indicating whether each transformer block should use the quantum implementation.
        If ``None`` all blocks are quantum.
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
        use_quantum_blocks: Optional[Iterable[bool]] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if use_quantum_blocks is None:
            use_quantum_blocks = [True] * num_blocks
        if len(use_quantum_blocks)!= num_blocks:
            raise ValueError("use_quantum_blocks length must match num_blocks")
        self.transformers = nn.Sequential(
            *[
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout=dropout)
                if flag
                else TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout=dropout)
                for flag in use_quantum_blocks
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

# --------------------------------------------------------------------------- #
# 4. Photonic fraud detection (strawberryfields)
# --------------------------------------------------------------------------- #

def build_fraud_detection_program(num_qubits: int = 2, clip: bool = True) -> "sf.Program":
    import strawberryfields as sf
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

    program = sf.Program(num_qubits)
    with program.context as q:
        BSgate(0.5, 0.0) | (q[0], q[1])
        for i in range(num_qubits):
            Rgate(0.1) | q[i]
            Sgate(0.2, 0.3) | q[i]
            Dgate(0.5, 0.0) | q[i]
            Kgate(0.1) | q[i]
    return program

__all__ = [
    "QLayer",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "HybridTextClassifier",
    "build_classifier_circuit",
    "EstimatorQNN_wrapper",
    "build_fraud_detection_program",
]
