"""Quantum‑augmented model using Pennylane for the variational encoder."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumEncoder(nn.Module):
    """
    Variational quantum circuit producing a 4‑dimensional output.

    Architecture
    ------------
    * 4 qubits encoded with 4‑bit general encoding (RYZXY pattern).
    * 3 layers of parameterized single‑qubit rotations and CNOT connectivity.
    * Measurement of Pauli‑Z expectation values on all qubits.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_wires)

        # Parameter shapes for the variational layers
        self.params = nn.Parameter(torch.randn(n_layers, n_wires, 3) * 0.1)

        # Encoder for classical input
        self.encoder = nn.Linear(n_wires, n_wires)

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Classical encoding (RYZXY)
        for i in range(self.n_wires):
            qml.RY(x[i], wires=i)
            qml.RZ(x[(i + 1) % self.n_wires], wires=i)
            qml.RX(x[(i + 2) % self.n_wires], wires=i)
            qml.RY(x[(i + 3) % self.n_wires], wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            for qubit in range(self.n_wires):
                qml.Rot(params[layer, qubit, 0],
                        params[layer, qubit, 1],
                        params[layer, qubit, 2],
                        wires=qubit)
            # CNOT ladder
            for qubit in range(self.n_wires - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, n_wires), expected to be normalized to [-1,1].

        Returns
        -------
        torch.Tensor
            Output of shape (B, n_wires) – Pauli‑Z expectation values.
        """
        batch_size = x.shape[0]
        out = torch.stack(
            [self._circuit(x[i], self.params) for i in range(batch_size)],
            dim=0,
        )
        return out


class QuantumNATEnhancedQ(nn.Module):
    """
    Hybrid model matching the interface of :class:`QuantumNATEnhanced`.

    Uses :class:`QuantumEncoder` to process the pooled feature vector.
    The rest of the forward pass mirrors the classical implementation.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            ResidualBlock(8, 16, stride=2),
            ResidualBlock(16, 32, stride=2),
        )
        self.encoder = QuantumEncoder()
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                     # (B,32,7,7)
        pooled = F.avg_pool2d(x, 6).view(x.size(0), -1)  # (B,32)
        out = self.encoder(pooled)                       # (B,4)
        return self.norm(out)


__all__ = ["QuantumEncoder", "QuantumNATEnhancedQ"]
