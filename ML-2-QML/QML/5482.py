"""
HybridTransformer – Quantum‑enabled implementation using TorchQuantum and Qiskit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler as Sampler


# --------------------------------------------------------------------------- #
# 1. Quantum building blocks
# --------------------------------------------------------------------------- #

class MultiHeadAttentionQuantum(nn.Module):
    """Multi‑head attention with quantum‑encoded projections."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.params, range(self.n_wires)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def _apply_q(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(batch, self.num_heads, self.d_k)
            head_outs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=1, device=head.device)
                head_outs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outs, dim=1))
        return torch.stack(projections, dim=1).view(batch, seq, self.embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = k = v = self._apply_q(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        return self.combine(out)


class FeedForwardQuantum(nn.Module):
    """Feed‑forward layer realized by a quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.params, range(self.n_qubits)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_qubits: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=1, device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
# 2. Transformer block and full quantum‑enabled model
# --------------------------------------------------------------------------- #

class TransformerBlockQuantum(nn.Module):
    """Transformer block with quantum attention and optional quantum feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class PositionalEncoder(nn.Module):
    """Same sinusoidal encoder as in the classical version."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                              (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class HybridTransformerQuantum(nn.Module):
    """Quantum‑enabled transformer with optional quantum autoencoder."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 8,
                 n_qubits_ffn: int = 8,
                 use_autoencoder: bool = False,
                 autoencoder_latent: int = 32,
                 autoencoder_hidden: Tuple[int, int] = (128, 64)):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                     n_qubits_transformer, n_qubits_ffn, dropout)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

        if use_autoencoder:
            self.autoencoder = AutoencoderQuantum(
                input_dim=embed_dim,
                latent_dim=autoencoder_latent,
                hidden_dims=autoencoder_hidden,
                dropout=dropout,
            )
        else:
            self.autoencoder = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        if self.autoencoder is not None:
            x = self.autoencoder(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# 3. Quantum autoencoder (based on Qiskit)
# --------------------------------------------------------------------------- #

class AutoencoderQuantum:
    """Quantum autoencoder using a swap‑test based ansatz."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.input_dim + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # ansatz
        qc.append(RealAmplitudes(self.latent_dim + self.input_dim, reps=5), range(self.latent_dim + self.input_dim))
        qc.barrier()

        # swap‑test
        aux = self.latent_dim + 2 * self.input_dim
        qc.h(aux)
        for i in range(self.input_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.input_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # placeholder: classical encoding for simplicity
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # placeholder: classical decoding
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# --------------------------------------------------------------------------- #
# 4. Classifier builder (quantum version)
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_qubits: int,
                             depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a variational circuit and return metadata."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, q in zip(encoding, range(num_qubits)):
        qc.rx(param, q)

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# 5. Quantum self‑attention (Qiskit)
# --------------------------------------------------------------------------- #

class QuantumSelfAttention:
    """Simple quantum self‑attention circuit."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build(self, rot: np.ndarray, ent: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rot[3 * i], i)
            qc.ry(rot[3 * i + 1], i)
            qc.rz(rot[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(ent[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> dict:
        qc = self._build(rotation_params, entangle_params)
        job = backend.run(qc, shots=shots)
        return job.result().get_counts(qc)


def SelfAttention() -> QuantumSelfAttention:
    return QuantumSelfAttention(n_qubits=4)


__all__ = [
    "HybridTransformerQuantum",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "AutoencoderQuantum",
    "build_classifier_circuit",
    "SelfAttention",
]
