"""Hybrid quantum model combining classical convolution, patch‑based quantum kernels,
and a measurement head.

The design fuses the QFCModel quantum module (encoder + random layer + measurement)
with the QuanvolutionFilter patch kernel.  We use a classical CNN to reduce the input
resolution, then apply a quantum kernel to each 2×2 patch of the feature map.
The measurement results are aggregated and passed through a classical linear
classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridNATModel(tq.QuantumModule):
    """Quantum hybrid model: classical conv + patch‑based quantum kernel."""

    class QuantumPatch(tq.QuantumModule):
        """Quantum kernel applied to a 2×2 image patch."""

        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            # Encode each pixel to a qubit via Ry
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )
            # Random circuit to entangle
            self.random = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random(qdev)
            return self.measure(qdev)

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.patch_size = 2
        # Quantum patch module
        self.qpatch = self.QuantumPatch()
        # Classical linear head after aggregation
        self.linear = nn.Linear(1, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        feat = self.features(x)  # [bsz, 16, H', W']
        _, c, h, w = feat.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0
        # Prepare quantum device
        n_wires = self.qpatch.n_wires
        qdev = tq.QuantumDevice(n_wires=n_wires, bsz=bsz, device=x.device, record_op=True)
        # Extract patches and apply quantum kernel
        measurements = []
        for r in range(0, h, self.patch_size):
            for col in range(0, w, self.patch_size):
                patch = feat[:, :, r : r + self.patch_size, col : col + self.patch_size]
                patch_flat = patch.view(bsz, -1)  # [bsz, 4]
                self.qpatch.encoder(qdev, patch_flat)
                meas = self.qpatch.forward(qdev)
                measurements.append(meas)
        # Concatenate and aggregate measurements
        out = torch.cat(measurements, dim=1)  # [bsz, num_patches]
        out = out.mean(dim=1, keepdim=True)  # [bsz, 1]
        out = self.norm(out)
        logits = self.linear(out)
        return logits


__all__ = ["HybridNATModel"]
