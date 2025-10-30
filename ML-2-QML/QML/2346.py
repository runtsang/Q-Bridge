"""Hybrid quantum model combining a classical encoder with a photonic‑inspired variational circuit."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer for the quantum circuit."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class HybridNATModel(tq.QuantumModule):
    """Hybrid quantum model with classical feature encoder and a photonic‑inspired variational layer."""

    class QLayer(tq.QuantumModule):
        def __init__(self, params: FraudLayerParameters):
            super().__init__()
            self.params = params
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            # Encode fraud parameters into rotation angles
            self.rx(qdev, wires=0, params=self.params.bs_theta)
            self.ry(qdev, wires=1, params=self.params.bs_phi)
            self.rz(qdev, wires=2, params=self.params.phases[0])
            self.crx(qdev, wires=[0, 3], params=self.params.phases[1])
            # Additional gates to emulate squeezing/displacement/kerr
            tqf.sx(qdev, wires=1, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[2, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(
        self,
        fraud_params: Sequence[FraudLayerParameters],
        n_wires: int = 4,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layers = nn.ModuleList([self.QLayer(p) for p in fraud_params])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.fc = nn.Linear(n_wires, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        for layer in self.q_layers:
            layer(qdev)
        out = self.measure(qdev)
        out = self.fc(out)
        return self.norm(out)

__all__ = ["HybridNATModel"]
