"""Quantum modules for the hybrid transformer.

Uses Pennylane to implement variational circuits for
attention projections and feed‑forward layers.  The
circuits are intentionally lightweight so that they can
be run on the local simulator or any supported backend.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn


class QuantumAttention(nn.Module):
    """Variational circuit that transforms a token vector."""

    def __init__(self, n_qubits: int, d_k: int, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.d_k = d_k
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        # Parameters for the ansatz
        self.params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3)
        )  # 3 parameters per qubit per layer

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """x: (batch, d_k) – encoded as rotation angles."""
        for i in range(self.n_qubits):
            qml.RX(x[:, i], wires=i)
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(params[l, i, 0], wires=i)
                qml.RZ(params[l, i, 1], wires=i)
                if i < self.n_qubits - 1:
                    qml.CNOT(wires=[i, i + 1])
            for i in range(self.n_qubits):
                qml.RX(params[l, i, 2], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, d_k) – token representation."""
        out = self.qnode(x, self.params)
        return torch.stack(out, dim=-1)


class QuantumFeedForward(nn.Module):
    """Variational circuit used as a feed‑forward layer."""

    def __init__(self, n_qubits: int, ffn_dim: int, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.ffn_dim = ffn_dim
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        self.params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3)
        )

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_qubits) – encoded as rotation angles."""
        for i in range(self.n_qubits):
            qml.RX(x[:, i], wires=i)
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(params[l, i, 0], wires=i)
                qml.RZ(params[l, i, 1], wires=i)
                if i < self.n_qubits - 1:
                    qml.CNOT(wires=[i, i + 1])
            for i in range(self.n_qubits):
                qml.RX(params[l, i, 2], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_qubits) – intermediate representation."""
        out = self.qnode(x, self.params)
        return torch.stack(out, dim=-1)


__all__ = ["QuantumAttention", "QuantumFeedForward"]
