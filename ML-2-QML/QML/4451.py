"""Hybrid quantum‑classical model inspired by Quantum‑NAT, QCNN, and kernel methods.

This module defines :class:`HybridQuantumNAT`, a torch‑quantum implementation that
combines a convolutional feature extractor with a simple rotation‑based kernel
ansatz and a sigmoid head.  It can be used as a drop‑in replacement for the
quantum part of the hybrid architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class HybridQuantumNAT(tq.QuantumModule):
    """Quantum module with convolutional feature extractor and rotation‑based kernel."""
    def __init__(self, n_wires: int = 4, shift: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 7 * 7, 64)
        self.shift = shift
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = [tq.RY(has_params=True, trainable=True) for _ in range(self.n_wires)]
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        angles = x[:, :self.n_wires]
        self.q_device.reset_states(bsz)
        for i, op in enumerate(self.ansatz):
            op(self.q_device, wires=i, params=angles[:, i])
        out = self.measure(self.q_device)
        out = out.view(bsz, -1)
        out = self.norm(out)
        probs = torch.sigmoid(out + self.shift)
        return probs

__all__ = ["HybridQuantumNAT"]
