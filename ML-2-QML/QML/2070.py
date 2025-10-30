"""Quantum‑NAT variant with an 8‑wire variational ansatz and classical‑quantum feature fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class VariationalLayer(tq.QuantumModule):
    """Deep parameterized ansatz on 8 wires with alternating RX, RZ, and CX gates."""

    def __init__(self):
        super().__init__()
        self.n_wires = 8
        self.rx = tq.RX(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.cx = tq.CX(trainable=False)  # fixed CX gates for entanglement

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Apply a two‑layer entangling block repeated 3 times
        for _ in range(3):
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.rz(qdev, wires=w)
            for i in range(0, self.n_wires - 1, 2):
                self.cx(qdev, wires=[i, i + 1])
            for i in range(1, self.n_wires - 1, 2):
                self.cx(qdev, wires=[i, i + 1])


class QuantumNATModel(tq.QuantumModule):
    """Quantum‑NAT model with a richer encoder and variational circuit, fused with classical features."""

    def __init__(self):
        super().__init__()
        self.n_wires = 8
        # Encoder using a 4×4 rotation‑based circuit (12 parameters) per wire
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"],
            n_wires=self.n_wires,
            n_layers=2,
        )
        self.qlayer = VariationalLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_wires + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(128, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )

        # Classical feature extraction: 64‑dim vector
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        classical = nn.functional.linear(pooled, torch.randn(64, 16, device=x.device))

        # Quantum processing
        self.encoder(qdev, pooled)
        self.qlayer(qdev)
        q_out = self.measure(qdev)  # shape (bsz, n_wires)

        # Fuse classical and quantum features
        fused = torch.cat([q_out, classical], dim=1)
        out = self.classifier(fused)
        return self.norm(out)


__all__ = ["QuantumNATModel"]
