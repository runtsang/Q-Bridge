"""Hybrid classical‑quantum binary classification: quantum counterpart.

This module implements the HybridQCNet class that mirrors the classical
counterpart but replaces the dense head with a quantum hybrid head
inspired by the Quantum‑NAT architecture.  The convolutional encoder is
identical, while the head maps the final feature to a 4‑wire quantum
device, measures, normalises, and then collapses to a binary logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """Parameterized 4‑wire layer from Quantum‑NAT."""

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)


class QuantumHybrid(nn.Module):
    """Quantum head that encodes the penultimate feature into 4 wires."""

    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(4)
        self.shift = shift
        self.fc = nn.Linear(4, 1)  # collapse to a single logit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bsz = inputs.shape[0]
        # Prepare quantum device
        qdev = tq.QuantumDevice(n_wires=4, bsz=bsz, device=inputs.device, record_op=True)
        # Reduce feature map to a 4‑dim vector
        pooled = F.avg_pool2d(inputs, kernel_size=6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        logits = self.fc(out)
        return logits


class HybridQCNet(nn.Module):
    """CNN encoder followed by a quantum hybrid head for binary classification."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional encoder identical to the classical version
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(0.5),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = QuantumHybrid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(inputs)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Pass through quantum hybrid head
        logits = self.hybrid(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridQCNet", "QuantumHybrid", "QLayer"]
