"""Quantum‑enhanced transformer classifier with optional quantum blocks.

All public API mirrors the classical implementation; a flag
``use_quantum=True`` activates quantum attention and feed‑forward
sub‑modules.  The module also provides a quantum regression dataset
compatible with the classical one, and a Qiskit‑based self‑attention
circuit for exploratory experiments.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data utilities
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (states, labels) where
    states are complex amplitudes of |0...0> and |1...1>
    superpositions.  Labels are sin(2θ)·cos(φ).
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis   = 2 * np.pi * np.random.rand(samples)

    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class QuantumRegressionDataset(Dataset):
    """Dataset providing quantum states and regression targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {"states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}


# --------------------------------------------------------------------------- #
# Qiskit self‑attention circuit
# --------------------------------------------------------------------------- #

class QuantumSelfAttention:
    """Simple Qiskit self‑attention block using rotations and CRX entanglement."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024):
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)


# --------------------------------------------------------------------------- #
# Quantum transformer blocks
# --------------------------------------------------------------------------- #

class MultiHeadAttentionQuantum(nn.Module):
    """Multi‑head attention where each linear projection is a quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
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

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 n_wires: int = 8):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_wires)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Project each token through a quantum circuit
        batch, seq, _ = x.shape
        proj = []
        for token in x.unbind(dim=1):  # shape: (batch, embed_dim)
            # split into heads
            heads = token.view(batch, self.num_heads, self.d_k)
            head_out = []
            for head in heads.unbind(dim=1):
                qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires,
                                        bsz=batch, device=head.device)
                head_out.append(self.q_layer(head, qdev))
            proj.append(torch.stack(head_out, dim=1))
        proj = torch.stack(proj, dim=1)  # (batch, seq, num_heads, d_k)
        # Standard scaled dot‑product
        attn_scores = torch.matmul(proj, proj.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, proj)
        attn_output = attn_output.transpose(1, 2).contiguous() \
                        .view(batch, seq, self.embed_dim)
        return self.combine(attn_output)


class FeedForwardQuantum(nn.Module):
    """Feed‑forward implemented via a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
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

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_wires: int,
                 dropout: float = 0.1):
        super().__init__()
        self.q_layer = self.QLayer(n_wires)
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    """Transformer block that can swap between quantum and classical sub‑modules."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_wires: int = 8,
                 use_quantum: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = (MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires)
                     if use_quantum else MultiHeadAttentionDense(embed_dim, num_heads, dropout))
        self.ffn = (FeedForwardQuantum(embed_dim, ffn_dim, n_wires, dropout)
                     if use_quantum else FeedForwardDense(embed_dim, ffn_dim, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Same sinusoidal encoder as in the classical version."""
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


class HybridTransformerClassifier(nn.Module):
    """Unified interface for classical and quantum transformer classifiers."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 n_wires: int = 8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        block_cls = (TransformerBlockQuantum if use_quantum else TransformerBlockDense)
        self.transformers = nn.Sequential(*[
            block_cls(embed_dim, num_heads, ffn_dim,
                      n_wires=n_wires, use_quantum=use_quantum,
                      dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "HybridTransformerClassifier",
    "QuantumRegressionDataset",
    "generate_superposition_data",
]
