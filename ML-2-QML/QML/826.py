"""Quantum quanvolution filter using a trainable Pennylane ansatz.

The circuit operates on 4 qubits (one per pixel in a 2×2 patch) and outputs a single expectation value
that serves as the patch embedding.  The ansatz is fully parameterised and trained jointly with the
classical network.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pennylane import numpy as np


class QuantumQuanvolutionFilter(nn.Module):
    """Parameterised quantum circuit that processes 2×2 image patches."""

    def __init__(self, n_wires: int = 4, n_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Define a device (CPU simulator)
        self.dev = qml.device("default.qubit", wires=self.n_wires)
        # Create a QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x):
            # Encode the 4 pixel values into Ry rotations
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i)
            # Parameterised layers
            for _ in range(self.n_layers):
                for i in range(self.n_wires):
                    qml.RZ(self.weight[i], wires=i)
                for i in range(self.n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Measure expectation of PauliZ on all qubits
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit
        # Trainable parameters
        self.weight = nn.Parameter(torch.randn(self.n_wires))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Patch tensor of shape (B, 4) where each row contains the 4 pixel values.

        Returns
        -------
        torch.Tensor
            Quantum embedding of shape (B, 1).
        """
        # Ensure input is float and on same device
        x = x.float()
        # Run the circuit in batch
        return self.circuit(x)


class QuanvolutionQuantumClassifier(nn.Module):
    """Hybrid network using the quantum quanvolution filter followed by a GRU head."""

    def __init__(
        self,
        in_channels: int = 1,
        patch_dim: int = 1,
        gru_hidden_dim: int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        # Since the quantum circuit outputs a single scalar per patch,
        # we reshape to (B, 196, 1) before feeding into the GRU
        self.gru = nn.GRU(
            input_size=patch_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(gru_hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W) where H=W=28.

        Returns
        -------
        torch.Tensor
            Log‑softmax over class scores, shape (B, num_classes).
        """
        # Extract 2×2 patches
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, C, 14, 14, 2, 2)
        patches = patches.contiguous().view(x.size(0), -1, 4)  # (B, 196, 4)
        # Apply quantum filter to each patch
        quantum_features = self.qfilter(patches)  # (B, 196, 1)
        # GRU expects (B, seq_len, feature_dim)
        _, h_n = self.gru(quantum_features)  # h_n: (1, B, gru_hidden_dim)
        h_n = h_n.squeeze(0)  # (B, gru_hidden_dim)
        logits = self.classifier(h_n)  # (B, num_classes)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuantumQuanvolutionFilter", "QuanvolutionQuantumClassifier"]
