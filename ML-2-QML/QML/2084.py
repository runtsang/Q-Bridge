"""Quantum quanvolution filter using a variational circuit and calibration layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np


class QuanvolutionFilter(nn.Module):
    """
    Quantum front‑end that maps 2×2 image patches to a 4‑dimensional feature vector.

    The filter encodes each pixel into a single qubit using RY rotations,
    applies a trainable StronglyEntanglingLayers ansatz, and measures the
    expectation value of Pauli‑Z on each wire.  A calibration linear layer
    maps the raw measurement results into a more expressive feature space.
    """

    def __init__(self, n_wires: int = 4, n_layers: int = 3, device: str = "default.qubit") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=n_wires)

        # Trainable parameters for the ansatz
        self.params = nn.Parameter(
            torch.randn(self.n_layers, self.n_wires, 3, dtype=torch.float32)
        )

        # Calibration layer maps raw Pauli‑Z expectations to a 4‑dimensional feature
        self.calibration = nn.Linear(n_wires, n_wires, bias=True)

        # Quantum node that executes the circuit
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Variational circuit.

        Args:
            inputs: Tensor of shape (batch, n_wires) containing RY angles.
            params: Tensor of shape (n_layers, n_wires, 3) for the ansatz.

        Returns:
            Tensor of shape (batch, n_wires) containing expectation values.
        """
        # Encode inputs into single‑qubit rotations
        for i in range(self.n_wires):
            qml.RY(inputs[:, i], wires=i)

        # Apply a trainable entangling layer
        qml.layers.StronglyEntanglingLayers(params, wires=range(self.n_wires))

        # Measure expectation value of Pauli‑Z on each wire
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 1, 28, 28).

        Returns:
            Flattened feature vector of shape (B, 4 * 14 * 14).
        """
        bsz = x.size(0)

        # Extract non‑overlapping 2×2 patches using unfold
        patches = (
            x.unfold(2, 2, 2)
           .unfold(3, 2, 2)
           .permute(0, 2, 3, 4, 5)
           .contiguous()
           .view(bsz, -1, 4)
        )  # shape: (B, 14*14, 4)

        # Process each patch through the quantum circuit
        features = []
        for i in range(patches.size(1)):
            # Shape of patch: (B, 4)
            patch = patches[:, i, :]
            # Quantum measurement → (B, 4)
            qout = self.qnode(patch, self.params)
            # Calibration → (B, 4)
            calibrated = self.calibration(qout)
            features.append(calibrated)

        # Concatenate all patch features → (B, 4*14*14)
        return torch.cat(features, dim=1)


class QuanvolutionClassifier(nn.Module):
    """
    Hybrid classifier that combines the quantum filter with a linear head.

    The architecture mirrors the classical counterpart: the filter
    extracts features which are then fed into a single dense layer.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 1, 28, 28).

        Returns:
            Log‑softmax probabilities of shape (B, num_classes).
        """
        features = self.qfilter(x)
        logits = self.linear(features)
        return torch.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
