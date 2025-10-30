"""Quantum fraud‑detection circuit using PennyLane for differentiable back‑propagation."""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (kept for API compatibility)."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionHybrid(nn.Module):
    """
    Variational quantum circuit that outputs a 2‑dimensional feature vector.
    The circuit is differentiable via PennyLane's PyTorch interface.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 3,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Trainable rotation parameters (theta, phi, lambda) for each qubit and layer
        self.params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3, dtype=torch.float32)
        )
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="backprop")
        if seed is not None:
            torch.manual_seed(seed)

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> List[torch.Tensor]:
        """Quantum circuit that maps a 2‑dimensional classical input to a 2‑dimensional feature vector."""
        # Encode input features into rotation angles
        for i in range(self.n_qubits):
            qml.RX(x[i], wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.Rot(
                    params[layer, q, 0],
                    params[layer, q, 1],
                    params[layer, q, 2],
                    wires=q,
                )
            # Entangling CNOT ladder
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

        # Return expectation values of Z for each qubit as the feature vector
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that computes quantum features.
        """
        # Ensure input shape (2,)
        x = x.view(self.n_qubits)
        return self.qnode(x, self.params)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
