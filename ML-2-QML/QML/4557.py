"""Quantum hybrid binary classifier that fuses CNN, quanvolution, and quantum kernel head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum 2x2 patch-based filter that applies a random two-qubit quantum kernel to image patches."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


class HybridQuantumBinaryClassifier(tq.QuantumModule):
    """Hybrid network using a CNN backbone, quantum quanvolution filter, and quantum kernel head."""

    def __init__(self, num_support: int = 10, gamma: float = 1.0) -> None:
        super().__init__()
        # CNN backbone adapted to accept 4 input channels from quanvolution
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.quanvolution = QuanvolutionFilter()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(375, 4)
        self.kernel = Kernel()
        self.support = nn.Parameter(torch.randn(num_support, 4))
        self.linear = nn.Linear(num_support, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convert to grayscale
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.quanvolution(x)  # shape: [batch, 784]
        x = x.view(x.size(0), 4, 14, 14)
        x = self.backbone(x)
        x = self.flatten(x)  # shape: [batch, 375]
        x = self.proj(x)  # shape: [batch, 4]
        # Compute quantum kernel between each sample and support vectors
        batch_size = x.shape[0]
        kernel_features = []
        for i in range(batch_size):
            xi = x[i]
            k_row = []
            for s in self.support:
                k_val = self.kernel(xi.unsqueeze(0), s.unsqueeze(0))
                k_row.append(k_val)
            kernel_features.append(torch.cat(k_row))
        kernel_features = torch.stack(kernel_features)
        logits = self.linear(kernel_features)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuanvolutionFilter", "Kernel", "HybridQuantumBinaryClassifier"]
