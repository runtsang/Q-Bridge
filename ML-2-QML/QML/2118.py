"""Quantum QuanvolutionClassifier with variational filter and confidence‑based early stopping."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionFilter(tq.QuantumModule):
    """Apply a variational circuit to each 2×2 image patch."""
    def __init__(self, n_wires: int = 4, n_layers: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.vqc = tq.RandomLayer(n_ops=n_layers, wires=list(range(n_wires)))
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
                self.vqc(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(tq.QuantumModule):
    """Quantum filter followed by a linear head. Supports confidence‑based early stopping."""
    def __init__(self, num_classes: int = 10, confidence_threshold: float = 0.9):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)
        self.confidence_threshold = confidence_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        probs = torch.exp(logits)
        max_conf, preds = probs.max(dim=1)
        mask = max_conf >= self.confidence_threshold
        final_preds = torch.where(mask, preds, torch.full_like(preds, -1))
        return final_preds

    def calibrate(self, temperature: float) -> None:
        self.linear.weight.data /= temperature
        self.linear.bias.data /= temperature


__all__ = ["QuanvolutionClassifier"]
