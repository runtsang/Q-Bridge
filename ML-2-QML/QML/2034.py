"""Quantum module for Quantum‑NAT with a variational circuit and classical post‑processing."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATGen240(tq.QuantumModule):
    """Quantum‑NAT model: encoder → variational circuit → measurement → classical head."""
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=60, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True, wires=0)
            self.ry = tq.RY(has_params=True, trainable=True, wires=1)
            self.rz = tq.RZ(has_params=True, trainable=True, wires=2)
            self.crx = tq.CRX(has_params=True, trainable=True, wires=[0, 3])
            self.cry = tq.CRY(has_params=True, trainable=True, wires=[1, 2])

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx(qdev)
            self.ry(qdev)
            self.rz(qdev)
            self.crx(qdev)
            self.cry(qdev)
            tqf.hadamard(qdev, wires=2)
            tqf.cnot(qdev, wires=[3, 0])

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.post_fc = nn.Sequential(
            nn.Linear(self.n_wires, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.post_fc(out)
        return self.norm(out)
