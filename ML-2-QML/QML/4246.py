"""Quantum hybrid model mirroring the classical counterpart, using torchquantum."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATHybrid(tq.QuantumModule):
    """Quantum version of the hybrid model, with encoder, variational layer, and classical classifier head."""
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self,
                 classifier_depth: int = 3,
                 num_features: int = 4,
                 ) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder mirrors the classical CNN: use a 4‑qubit encoder
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical classifier head
        self.classifier = self._build_classifier(num_features, classifier_depth)
        self.norm = nn.BatchNorm1d(2)

    def _build_classifier(self, num_features: int, depth: int) -> nn.Sequential:
        layers = []
        in_dim = self.n_wires
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical pooling to produce a 4‑dim vector per sample
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode classical features into qubits
        self.encoder(qdev, pooled)
        # Variational layer
        self.q_layer(qdev)
        # Measurement
        out = self.measure(qdev)  # shape (bsz, n_wires)
        # Classical classifier
        logits = self.classifier(out)
        return self.norm(logits)

__all__ = ["QuantumNATHybrid"]
