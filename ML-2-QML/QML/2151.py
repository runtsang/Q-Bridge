"""Quantum-enhanced quanvolution using Pennylane variational circuits."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np


class QuantumDepthwiseQuanvolutionFilter(nn.Module):
    """Depth‑wise separable quanvolution implemented with a Pennylane QNode per patch."""
    def __init__(self, in_channels: int = 1, n_qubits: int = 4,
                 depth: int = 2, dropout: float = 0.1, residual: bool = True,
                 device: str | qml.Device = "default.qubit") -> None:
        super().__init__()
        self.residual = residual
        self.dropout = nn.Dropout2d(dropout)
        self.n_qubits = n_qubits

        # Define a parameterised ansatz for each patch
        self.qnode = qml.QNode(self._quantum_circuit, backend=device,
                               interface="torch", diff_method="parameter-shift")

        # Point‑wise linear mixer after the quantum block
        self.pointwise = nn.Conv2d(in_channels, in_channels * 4, kernel_size=1,
                                   stride=1, bias=True)

    def _quantum_circuit(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Variational circuit: encode input, apply entangling layers, measure Z."""
        # Encode four input values into RY rotations
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)

        # Entangling layers
        for layer in range(self.qnode.n_layers):
            for i in range(self.n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # Measure expectation of PauliZ on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        assert channels == 1, "Only single‑channel input supported"

        # Prepare patches (non‑overlapping 2×2)
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # shape: (N, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(batch, 14, 14, 4)  # (N, 14, 14, 4)
        patches = patches.permute(0, 3, 1, 2)  # (N, 4, 14, 14)

        # Apply quantum circuit to each patch
        outputs = torch.zeros_like(patches)
        for n in range(batch):
            for h in range(14):
                for w in range(14):
                    inp = patches[n, :, h, w]
                    out = self.qnode(inp)  # shape (4,)
                    outputs[n, :, h, w] = out

        # Dropout on quantum outputs
        outputs = self.dropout(outputs)

        # Point‑wise mixing
        pw = self.pointwise(outputs)

        # Residual connection
        if self.residual:
            # Upsample original input to match output size
            residual = F.interpolate(x, size=pw.shape[2:], mode='nearest')
            return pw + residual
        return pw


class QuanvolutionPlusClassifier(nn.Module):
    """Full QML classifier mirroring the classical variant."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuantumDepthwiseQuanvolutionFilter()
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.classifier(features.view(x.size(0), -1))
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuantumDepthwiseQuanvolutionFilter", "QuanvolutionPlusClassifier"]
