"""
Hybrid quantum kernel module with a variational embedding and optional classical preprocessing.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from typing import Sequence, Optional

# Number of qubits used in the variational circuit
DEV_WIRES = 4
dev = qml.device("default.qubit", wires=DEV_WIRES)

def _variational_circuit(x: np.ndarray) -> None:
    """
    Encode a real vector x into a quantum state using Ry rotations
    followed by a simple linear‑chain CNOT entanglement.
    """
    for i, xi in enumerate(x):
        qml.RY(xi, wires=i)
    for i in range(DEV_WIRES - 1):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev)
def _state_circuit(x: np.ndarray) -> np.ndarray:
    _variational_circuit(x)
    return qml.state()

class VariationalEmbedding(nn.Module):
    """
    Classical feed‑forward network that reduces dimensionality before
    the quantum embedding.  It is fully differentiable and can be
    trained jointly with a kernel alignment loss.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class KernalAnsatz:
    """
    Quantum kernel using a variational ansatz.  An optional classical
    embedding can be supplied which is applied before the quantum
    circuit.
    """
    def __init__(self, embedding: Optional[VariationalEmbedding] = None):
        self.embedding = embedding

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.embedding is not None:
            # Convert raw data to tensors, embed, and convert back to numpy
            x_t = torch.tensor(x, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            x_emb = self.embedding(x_t).detach().numpy()
            y_emb = self.embedding(y_t).detach().numpy()
            x, y = x_emb, y_emb
        state_x = _state_circuit(x)
        state_y = _state_circuit(y)
        overlap = np.vdot(state_x, state_y)
        return float(np.abs(overlap) ** 2)

class Kernel:
    """
    Wrapper that mimics the original API.  It exposes a ``forward`` method
    that accepts 1‑D arrays and returns the kernel value.
    """
    def __init__(self, embedding: Optional[VariationalEmbedding] = None):
        self.ansatz = KernalAnsatz(embedding)

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        x = x.reshape(1, -1).flatten()
        y = y.reshape(1, -1).flatten()
        return self.ansatz(x, y)

def kernel_matrix(a: Sequence[np.ndarray],
                  b: Sequence[np.ndarray],
                  embedding: Optional[VariationalEmbedding] = None) -> np.ndarray:
    """
    Compute the Gram matrix for two collections of samples using the quantum kernel.
    """
    kernel = Kernel(embedding)
    return np.array([[kernel(x, y) for y in b] for x in a])

def train_embedding(a: Sequence[np.ndarray],
                    b: Sequence[np.ndarray],
                    out_dim: int,
                    epochs: int = 200,
                    lr: float = 1e-3) -> VariationalEmbedding:
    """
    Train a lightweight classical embedding so that the induced quantum kernel
    aligns with an identity target matrix.  The loss is the Frobenius norm
    between the Gram matrix and the identity.
    """
    in_dim = a[0].shape[-1]
    emb = VariationalEmbedding(in_dim, out_dim)
    opt = torch.optim.Adam(emb.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        km = kernel_matrix(a, b, embedding=emb)
        target = np.eye(len(a))
        loss = torch.tensor(np.linalg.norm(km - target), dtype=torch.float32, requires_grad=True)
        loss.backward()
        opt.step()
    return emb

__all__ = ["VariationalEmbedding", "KernalAnsatz", "Kernel", "kernel_matrix",
           "train_embedding"]
