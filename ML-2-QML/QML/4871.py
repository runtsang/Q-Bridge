"""Quantum hybrid network that applies a quanvolution circuit and a linear head.

The module integrates:
- A quantum quanvolution filter that encodes 2×2 image patches into 4‑qubit states.
- A linear readout head, trained classically.
- Optional Gaussian shot‑noise to emulate measurement uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum 2×2 patch encoder followed by a random layer and Pauli‑Z measurement."""
    def __init__(self, n_wires: int = 4, stride: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.stride = stride
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        # reshape to 28×28 images
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, self.stride):
            for c in range(0, 28, self.stride):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # create a fresh device for this batch of patches
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, data)
                self.layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        return torch.cat(patches, dim=1)


class QuantumQuanvolutionNet(tq.QuantumModule):
    """
    Hybrid quantum‑classical network:
    quantum quanvolution filter → linear readout.
    Optionally adds Gaussian shot noise to the final logits.
    """
    def __init__(
        self,
        n_wires: int = 4,
        num_classes: int = 10,
        use_noise: bool = False,
        shots: int | None = None,
    ):
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter(n_wires=n_wires)
        self.linear = nn.Linear(n_wires * 14 * 14, num_classes)
        self.use_noise = use_noise
        self.shots = shots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        if self.use_noise and self.shots is not None:
            rng = np.random.default_rng()
            noise = rng.normal(0, 1 / np.sqrt(self.shots), size=logits.shape)
            logits = logits + torch.tensor(noise, dtype=logits.dtype, device=logits.device)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuantumQuanvolutionFilter", "QuantumQuanvolutionNet"]
