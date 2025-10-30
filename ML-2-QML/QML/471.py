"""Quantum hybrid model with a variational quanvolution filter.

The quantum branch implements a parameterized two‑qubit circuit
applied to each 2×2 patch of the input image.  A trainable variational
layer replaces the random layer of the seed, allowing the model to learn
task‑specific quantum features.  The outputs are concatenated with a
classical linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class VariationalQuanvolutionFilter(tq.QuantumModule):
    """Apply a trainable two‑qubit variational circuit to 2×2 image patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder mapping pixel intensities to qubit rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable variational layer
        self.var_layer = tq.VariationalLayer(
            n_ops=8,
            wires=list(range(self.n_wires)),
            params_init="random",
            param_shapes=[(1,)] * 8,
        )
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
                self.var_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuanvolutionGen104(tq.QuantumModule):
    """Hybrid quantum classifier that uses a variational quanvolution filter
    followed by a classical linear head."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = VariationalQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen104"]
