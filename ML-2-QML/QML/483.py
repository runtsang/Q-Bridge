"""Quanvolution network using a trainable quantum kernel and depth‑controlled variational layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionNet(tq.QuantumModule):
    """Hybrid network: quantum kernel over 2×2 patches followed by a classical linear head."""
    def __init__(self, depth: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.n_wires = 4
        # Feature‑map encoder: map 4 pixel values to qubits using Ry rotations
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        # Trainable variational layers
        self.q_layers = nn.ModuleList()
        for _ in range(depth):
            self.q_layers.append(
                tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)), trainable=True)
            )
        # Measurement of all qubits
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical classifier head
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        # Reshape input to 28×28 patches
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
                for layer in self.q_layers:
                    layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        # Concatenate all patch features
        features = torch.cat(patches, dim=1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
