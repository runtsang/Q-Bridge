"""Quantum‑augmented model using Pennylane and Torch.

The circuit:
    * Encodes 16‑dimensional pooled features into a 4‑wire register
      using RX rotations.
    * Applies *n_layers* variational layers each consisting of
      parameterised rotations (RZ, RX, RY) followed by CNOT
      entanglement in a ring topology.
    * Measures Pauli‑Z expectation values on each wire.
    * Normalises the output with a learnable BatchNorm1d.

The class is fully Torch‑compatible and can be trained with any
gradient‑based optimiser via the autograd machinery provided by
Pennylane.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as np


class QFCModel(nn.Module):
    """Variational quantum fully‑connected model.

    Parameters
    ----------
    n_wires : int
        Number of qubits (default: 4).
    n_layers : int
        Number of variational layers (default: 3).
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_wires, shots=None)

        # Variational parameters: shape (n_layers, n_wires, 3) for (RZ, RX, RY)
        self.vars = nn.Parameter(torch.randn(self.n_layers, self.n_wires, 3) * np.pi)

        # BatchNorm for the 4‑dimensional output
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Wrap the circuit with a QNode so autograd works
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Quantum circuit applied to a single data point.

        Parameters
        ----------
        x : torch.Tensor
            16‑dimensional feature vector (after pooling).
        params : torch.Tensor
            Variational parameters of shape (n_layers, n_wires, 3).

        Returns
        -------
        torch.Tensor
            Expected Pauli‑Z values for each wire.
        """
        # Feature encoding: RX with the 16 features (slice into 4 groups)
        for i in range(self.n_wires):
            qml.RX(x[i * 4 + 0], wires=i)
            qml.RX(x[i * 4 + 1], wires=i)
            qml.RX(x[i * 4 + 2], wires=i)
            qml.RX(x[i * 4 + 3], wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            for w in range(self.n_wires):
                qml.RZ(params[layer, w, 0], wires=w)
                qml.RX(params[layer, w, 1], wires=w)
                qml.RY(params[layer, w, 2], wires=w)
            # Ring entanglement
            for w in range(self.n_wires):
                qml.CNOT(wires=[w, (w + 1) % self.n_wires])

        # Expectation values of Pauli‑Z on each wire
        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Normalised 4‑dimensional output.
        """
        bsz = x.shape[0]
        # Average‑pool to 16 features per sample
        pooled = torch.nn.functional.avg_pool2d(x, kernel_size=6).view(bsz, 16)

        # Run the QNode in vectorised mode
        out = self.qnode(pooled, self.vars)  # shape (B, n_wires)
        return self.norm(out)


__all__ = ["QFCModel"]
