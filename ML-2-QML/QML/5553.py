"""
Quantum‑enhanced self‑attention layer using Pennylane.
The class mirrors the classical hybrid module but implements the
attention kernel as a variational circuit.  It can be used as a drop‑in
replacement for the classical SelfAttention in a quantum‑classical
pipeline.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Callable, Tuple, Any

import pennylane as qml
import numpy as np
import torch
from torch import nn

# Import the classical hybrid module to reuse its logic
from.SelfAttention__gen457 import HybridSelfAttention

# Quantum device
DEV = qml.device("default.qubit", wires=4)

# Variational circuit for attention
def _attention_circuit(
    rotation_params: np.ndarray,
    entangle_params: np.ndarray,
    n_qubits: int = 4,
) -> np.ndarray:
    """Return a vector of expectation values that will be used as attention scores."""
    @qml.qnode(DEV)
    def circuit():
        for i in range(n_qubits):
            qml.RX(rotation_params[3 * i], wires=i)
            qml.RY(rotation_params[3 * i + 1], wires=i)
            qml.RZ(rotation_params[3 * i + 2], wires=i)
        for i in range(n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return np.array(circuit())

# Quantum‑enhanced self‑attention module
class QuantumSelfAttention(nn.Module):
    """Variational‑circuit based self‑attention.  The attention scores are
    obtained from the expectation values of Pauli‑Z measurements on a
    4‑qubit circuit."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        # Classical projections
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        # Quantum attention scores
        scores_np = _attention_circuit(rotation_params, entangle_params)
        scores = torch.as_tensor(scores_np, dtype=torch.float32, device=inputs.device)
        scores = scores.unsqueeze(0).expand(inputs.size(0), -1)

        # Weighted sum
        out = torch.matmul(scores, v)
        return out

# Wrapper that forwards to the hybrid version when quantum is disabled
class QuantumHybridAttention(HybridSelfAttention):
    """Combines the classical hybrid attention with a Pennylane variational
    circuit when the user supplies rotation and entangle parameters."""
    def __init__(self, embed_dim: int = 4, n_qubits: int = 0):
        super().__init__(embed_dim, n_qubits)
        if n_qubits > 0:
            self.quantum = QuantumSelfAttention(embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> torch.Tensor:
        if self.quantum and rotation_params is not None and entangle_params is not None:
            return self.quantum(inputs, rotation_params, entangle_params)
        return super().forward(inputs, rotation_params, entangle_params)

__all__ = ["QuantumSelfAttention", "QuantumHybridAttention"]
