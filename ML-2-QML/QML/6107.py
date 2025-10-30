"""Hybrid model with classical backbone and a quantum fully‑connected layer, inspired by Quanvolution and Quantum‑NAT."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuanvolutionHybridModel(tq.QuantumModule):
    """Classical CNN backbone + quantum layer + classifier."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4

        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Projection to quantum register size
        self.proj = nn.Linear(32, self.n_wires)

        # Quantum encoder and layer
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Normalisation and classification
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.classifier = nn.Linear(self.n_wires, 10)

    class QLayer(tq.QuantumModule):
        """Parameterised quantum sub‑module."""
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:  # type: ignore[override]
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        # Classical feature extraction
        features = self.features(x)
        flattened = features.view(bsz, -1)
        # Project to quantum register
        projected = self.proj(flattened)

        # Quantum processing
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, projected)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)

        # Classification head
        logits = self.classifier(out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybridModel"]
