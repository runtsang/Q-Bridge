"""Quanvolution filter using a parameterized variational quantum circuit."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionFilter(tq.QuantumModule):
    """Apply a learnable two‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps pixel intensities to rotation angles
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Parameterized variational circuit
        self.circuit = tq.Circuit(n_wires)
        # Register trainable parameters
        self.ry0 = nn.Parameter(torch.rand(1))
        self.ry1 = nn.Parameter(torch.rand(1))
        self.ry2 = nn.Parameter(torch.rand(1))
        self.ry3 = nn.Parameter(torch.rand(1))
        for _ in range(depth):
            self.circuit += tq.ry(0, param=self.ry0)
            self.circuit += tq.ry(1, param=self.ry1)
            self.circuit += tq.ry(2, param=self.ry2)
            self.circuit += tq.ry(3, param=self.ry3)
            self.circuit += tq.cnot(0, 1)
            self.circuit += tq.cnot(2, 3)
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
                self.circuit(qdev)  # apply parameterized circuit
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the variational quanvolution filter."""
    def __init__(self, in_features: int = 4 * 14 * 14, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
