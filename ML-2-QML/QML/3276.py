"""Quantum‑enhanced transformer based on Qiskit.

The implementation follows the same public constructor and
forward signature as the classical variant.  The class accepts
rotation_params and entangle_params which are used by the quantum
attention circuit.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_attention_circuit(
    n_qubits: int,
    rotation_params: np.ndarray,
    entangle_params: np.ndarray,
) -> QuantumCircuit:
    """
    Build a simple attention‑style circuit that applies a rotation per qubit
    followed by a ring of CNOT gates and measurement.  The circuit returns
    expectation values of Pauli‑Z for each qubit.
    """
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    circuit = QuantumCircuit(qr, cr)

    # Rotation layer
    for i in range(n_qubits):
        circuit.rx(rotation_params[i], qr[i])
        circuit.ry(rotation_params[i + n_qubits], qr[i])
        circuit.rz(rotation_params[i + 2 * n_qubits], qr[i])

    # Entanglement layer (ring of CNOTs)
    for i in range(n_qubits - 1):
        circuit.cx(qr[i], qr[i + 1])
    circuit.cx(qr[-1], qr[0])

    # Entanglement gate that depends on entangle_params
    for i in range(n_qubits - 1):
        circuit.rz(entangle_params[i], qr[i + 1])

    circuit.measure(qr, cr)
    return circuit


class QuantumSelfAttention(nn.Module):
    """
    Maps a batch of token embeddings to a batch of quantum expectation values.
    The circuit is executed on a local Aer simulator.
    """

    def __init__(self, n_qubits: int, device: str = "qasm_simulator"):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend(device)

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """
        x: (B, T, embed_dim) – input token embeddings
        Returns: (B, T, n_qubits) – expectation values of Pauli‑Z
        """
        batch, seq, _ = x.size()
        out = torch.empty(batch, seq, self.n_qubits, device=x.device)
        for b in range(batch):
            for t in range(seq):
                circuit = _build_attention_circuit(
                    self.n_qubits, rotation_params, entangle_params
                )
                job = execute(circuit, self.backend, shots=1024)
                result = job.result()
                counts = result.get_counts(circuit)
                # Convert counts to expectation values
                exp = np.zeros(self.n_qubits)
                for bitstring, cnt in counts.items():
                    bits = np.array([int(b) for b in bitstring[::-1]])
                    exp += cnt * (1 - 2 * bits)  # +1 for 0, -1 for 1
                exp = exp / 1024.0
                out[b, t] = torch.from_numpy(exp).to(x.device)
        return out


class QuantumFeedForward(nn.Module):
    """
    Small feed‑forward network implemented as a quantum circuit that
    produces expectation values of Pauli‑Z for each qubit.
    """

    def __init__(self, n_qubits: int, device: str = "qasm_simulator"):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, n_qubits) – input from attention
        Returns: (B, T, n_qubits) – processed output
        """
        batch, seq, _ = x.size()
        out = torch.empty(batch, seq, self.n_qubits, device=x.device)
        for b in range(batch):
            for t in range(seq):
                circuit = QuantumCircuit(self.n_qubits)
                for i in range(self.n_qubits):
                    circuit.rx(np.random.uniform(0, 2 * np.pi), i)
                circuit.measure_all()
                job = execute(circuit, self.backend, shots=1024)
                result = job.result()
                counts = result.get_counts(circuit)
                exp = np.zeros(self.n_qubits)
                for bitstring, cnt in counts.items():
                    bits = np.array([int(b) for b in bitstring[::-1]])
                    exp += cnt * (1 - 2 * bits)
                exp = exp / 1024.0
                out[b, t] = torch.from_numpy(exp).to(x.device)
        return out


class QuantumTransformerBlock(nn.Module):
    """
    One block of the quantum transformer.  It first applies the quantum
    self‑attention, then a quantum feed‑forward, and finally a classical
    residual connection with a learnable linear projection.
    """

    def __init__(
        self,
        n_qubits: int,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = QuantumSelfAttention(n_qubits)
        self.ffn = QuantumFeedForward(n_qubits)
        self.proj = nn.Linear(n_qubits, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        attn_out = self.attn(x, rotation_params, entangle_params)
        ffn_out = self.ffn(attn_out)
        proj = self.proj(ffn_out)
        return self.norm(self.dropout(x + proj))


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-np.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class UnifiedSelfAttentionTransformer(nn.Module):
    """
    Quantum‑enhanced transformer that shares the same public constructor and
    forward signature as the classical variant.  The class accepts
    rotation_params and entangle_params which are used by the quantum
    attention circuit.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_blocks: int,
        dropout: float = 0.1,
        n_qubits: int = 4,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList(
            [
                QuantumTransformerBlock(n_qubits, embed_dim, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)  # binary classification

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x, rotation_params, entangle_params)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling
        return self.classifier(x)
