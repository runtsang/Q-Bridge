"""Hybrid quantum-classical quanvolution with PennyLane."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np  # noqa: F401

class QuantumPatchEncoder(nn.Module):
    """Parameter‑efficient quantum circuit that maps a 2×2 patch to 4 expectation values."""
    def __init__(self, num_wires: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=self.num_wires)
        self.params = nn.Parameter(torch.randn(self.num_layers, self.num_wires, 2))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            # Encode input into Ry rotations
            for i in range(self.num_wires):
                qml.RY(x[i], wires=i)
            # Variational layers
            for layer in range(self.num_layers):
                for i in range(self.num_wires):
                    qml.RZ(params[layer, i, 0], wires=i)
                    qml.RX(params[layer, i, 1], wires=i)
                # Entangling CNOT chain
                for i in range(self.num_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.num_wires - 1, 0])  # wrap‑around
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_wires)
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(self.circuit(x[i], self.params))
        return torch.stack(outputs, dim=0)

class QuanvolutionAdvanced(nn.Module):
    """Hybrid quantum‑classical model: quantum patch encoder, mean aggregation, linear head."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.encoder = QuantumPatchEncoder()
        self.classifier = nn.Linear(self.encoder.num_wires, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28)
        Returns:
            log probabilities: Tensor of shape (batch, num_classes)
        """
        batch_size = x.shape[0]
        patch_list = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r + 2, c:c + 2]  # (batch, 2, 2)
                patch = patch.view(batch_size, -1)  # (batch, 4)
                patch_list.append(patch)
        patches = torch.stack(patch_list, dim=1)  # (batch, num_patches, 4)
        encoded = []
        for i in range(patches.shape[1]):
            encoded.append(self.encoder(patches[:, i, :]))
        encoded = torch.stack(encoded, dim=1)  # (batch, num_patches, 4)
        context = encoded.mean(dim=1)  # (batch, 4)
        logits = self.classifier(context)  # (batch, num_classes)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionAdvanced"]
