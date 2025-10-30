import pennylane as qml
import torch
from pennylane import numpy as np
from pennylane import qnode
from pennylane import device
from pennylane import circuits
from pennylane import ops
from typing import Tuple

class QuantumNATEnhanced:
    """
    Quantum‑only implementation of the hybrid model’s variational block.
    Uses a 4‑wire PennyLane device with a parameter‑efficient circuit:
        • 20 two‑qubit entangling gates (CRX, CNOT) and single‑qubit rotations.
        • Parameter sharing across layers for reduced trainable parameters.
        • Supports both simulator (default) and real device back‑ends.
    """

    def __init__(self, n_wires: int = 4, n_layers: int = 4, device_name: str = "default.qubit") -> None:
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = device(device_name, wires=n_wires)
        self.qnode = qnode(self._circuit, device=self.dev, interface="torch")

    def _circuit(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        # Encode classical data into rotation angles
        for i in range(self.n_wires):
            ops.RY(x[i % x.shape[0]], wires=i)
        # Variational layers with shared parameters
        for _ in range(self.n_layers):
            for i in range(self.n_wires):
                ops.RX(params[i], wires=i)
                ops.RZ(params[(i + 1) % self.n_wires], wires=i)
            for i in range(self.n_wires - 1):
                ops.CRX(params[i], wires=[i, i + 1])
                ops.CNOT(wires=[i, i + 1])
        # Measurement
        return ops.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of data.
        Parameters
        ----------
        x : torch.Tensor
            Batch of shape (B, 4) where each row contains 4 classical features.
        Returns
        -------
        torch.Tensor
            Batch of shape (B, 1) containing expectation values.
        """
        B = x.shape[0]
        # Generate random parameters for demonstration; in practice learnable.
        params = np.random.uniform(0, 2 * np.pi, size=(self.n_wires,))
        out = torch.stack([self.qnode(params, x[i].numpy()) for i in range(B)])
        return out

__all__ = ["QuantumNATEnhanced"]
