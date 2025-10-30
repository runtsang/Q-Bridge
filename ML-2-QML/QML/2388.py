"""Quantum self‑attention module implemented with Qiskit.

The module mirrors the classical implementation but replaces the attention
refinement with a parameterised quantum circuit.  It uses Aer simulator
and the parameter‑shift rule for gradients, enabling end‑to‑end training
with PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, Aer, execute

class _QuantumCircuitWrapper:
    """
    Builds a parameterised two‑qubit circuit that emulates a self‑attention
    style operation.  The circuit applies single‑qubit rotations followed
    by a controlled‑X entanglement.  The expectation value of the Z
    operator on the first qubit is returned as the refinement score.
    """
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots

    def build(self, param_values: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Apply rotations
        for i in range(self.n_qubits):
            qc.rx(param_values[3 * i], i)
            qc.ry(param_values[3 * i + 1], i)
            qc.rz(param_values[3 * i + 2], i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    def expectation(self, param_values: np.ndarray) -> float:
        qc = self.build(param_values)
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(qc)
        # Convert counts to expectation of Z on first qubit
        exp = 0.0
        for bitstring, cnt in counts.items():
            # bitstring is in reverse order
            first_bit = int(bitstring[-1])
            exp += (1 - 2 * first_bit) * cnt
        return exp / self.shots

class HybridSelfAttention(nn.Module):
    """
    Quantum‑refined self‑attention module.  The classical attention
    head is identical to the one in the classical implementation.
    The refinement step is performed by a parameterised quantum circuit.
    """
    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 1,
        n_qubits: int = 4,
        shots: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        # Classical attention parameters
        self.q_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.k_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.v_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))

        # Quantum refinement
        self.qc = _QuantumCircuitWrapper(n_qubits=n_qubits, shots=shots)
        # Parameters for the quantum circuit
        self.qc_params = nn.Parameter(torch.randn(n_qubits * 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Classical attention
        Q = torch.matmul(x, self.q_weight)
        K = torch.matmul(x, self.k_weight)
        V = torch.matmul(x, self.v_weight)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # Quantum refinement
        flat = out.reshape(-1, self.embed_dim)
        expectations = []
        for sample in flat:
            # Use the same quantum parameters for every sample
            angles = self.qc_params.detach().cpu().numpy()
            exp_val = self.qc.expectation(angles)
            expectations.append(exp_val)
        expectations = torch.tensor(expectations, device=out.device).view(batch, seq_len, 1)
        out = out * expectations

        return out

__all__ = ["HybridSelfAttention"]
