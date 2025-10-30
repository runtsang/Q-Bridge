"""CompositeSelfAttentionQNN – classical backbone with quantum‑enhanced attention."""

from __future__ import annotations

import itertools
import math
from typing import Iterable, Tuple, Iterable

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Classical attention backbone
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """
    A multi‑head, layer‑norm‑based self‑attention module that mirrors the interface
    used by the quantum self‑attention helper while adding depth, dropout and
    residual connections.  The module accepts *rotation* and *entangle* parameters
    that are **not** used in the classical path – they are kept for API
    compatibility with the quantum counterpart.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        # The rotation/entangle params are unused; they are accepted for API parity.
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        # Reshape for multi‑head
        q = q.view(-1, inputs.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, inputs.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, inputs.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(-1, inputs.size(1), self.embed_dim)
        out = self.out_proj(out)
        return self.ln(out + inputs)


# --------------------------------------------------------------------------- #
#  Quantum‑enhanced attention block
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(nn.Module):
    """
    Wraps a variational Qiskit circuit that produces an attention‑style probability
    distribution over the sequence length.  The circuit uses a rotation‑gate
    ansatz followed by an entangling layer; its measurement counts are turned
    into a softmax probability vector.  The module can be run on any
    qiskit Aer backend or a real device.
    """

    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._build_circuit()

    def _build_circuit(self):
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        # rotation layer
        for i in range(self.n_qubits):
            self.circuit.rx(0.0, i)
            self.circuit.ry(0.0, i)
            self.circuit.rz(0.0, i)
        # entanglement layer
        self.circuit.cx(self.n_qubits - 1, self.n_qubits - 1)  # placeholder

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        # Inject parameters into the circuit
        for i in range(self.n_qubits):
            self.circuit.rx(rotation_params[3 * i], i)
            self.circuit.ry(rotation_params[3 * i + 1], i)
            self.circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            self.circuit.cnot(entangle_params[i], i, i + 1)

        # Execute and convert counts to probability distribution
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots)
        counts = job.result().get_counts(self.circuit)
        probs = np.zeros(self.n_qubits)
        for bitstring, c in counts.items():
            idx = int(bitstring[::-1], 2)
            probs[idx] = c / self.shots
        probs = torch.tensor(probs, dtype=torch.float32)
        # Map the attention scores to the sequence length
        # Assume the inputs have shape (batch, seq_len, dim)
        seq_len = inputs.shape[1]
        if seq_len!= self.n_qubits:
            # Pad or truncate
            probs = probs[:seq_len] if seq_len < self.n_qubits else probs[:seq_len]
        # Broadcast to match batch dimension
        probs = probs.unsqueeze(0).repeat(inputs.shape[0], 1)
        # Apply attention to inputs (element‑wise weighting)
        return torch.mul(inputs, probs.unsqueeze(-1))


# --------------------------------------------------------------------------- #
#  Hybrid self‑attention network
# --------------------------------------------------------------------------- #
class CompositeSelfAttentionQNN(nn.Module):
    """
    End‑to‑end model that ingests embeddings, passes them through a classical
    self‑attention block and a quantum‑enhanced block, and merges the outputs
    using a weighted sum.  A graph‑based fidelity adjacency is built from
    the hidden states of both modalities to enable cross‑validation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        n_qubits: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.classical = ClassicalSelfAttention(embed_dim, num_heads, dropout)
        self.quantum = QuantumSelfAttention(n_qubits, shots=1024)
        self.alpha = alpha
        self.fidelity_graph = None

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, nx.Graph]:
        # Classical attention output
        class_out = self.classical(rotation_params, entangle_params, inputs)

        # Quantum attention output
        quantum_out = self.quantum(rotation_params, entangle_params, inputs)

        # Merge via weighted sum
        out = self.alpha * class_out + (1 - self.alpha) * quantum
        return out, self._build_fidelity_graph(class_out, quantum_out)

    def _build_fidelity_graph(self, class_state: torch.Tensor, quantum_state: torch.Tensor) -> nx.Graph:
        # Flatten batch‑wise for graph construction
        flat_class = class_state.reshape(-1, class_state.shape[-1])
        flat_q = quantum_state.reshape(-1, quantum_state.shape[-1])
        # Compute pairwise dot products (fidelity-like)
        fid_matrix = torch.einsum('ij,ij->i', flat_class, flat_q)
        graph = nx.Graph()
        for idx, fid in enumerate(fid_matrix):
            graph.add_node(idx, fidelity=float(fid))
        # Each pair connects with weight ~ (i * 0..1)
        for i, j in itertools.combinations(range(fid_matrix.shape[0]), 2):
            weight = 1 - abs(fid_matrix[i] - fid_matrix[j])
            graph.add_edge(i, j, weight=weight)
        return graph


__all__ = ["CompositeSelfAttentionQNN"]
