"""Hybrid quantum model mirroring the classical fraud‑detection structure with a variational circuit."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (kept for symmetry)."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class FraudQuantumLayer(tq.QuantumModule):
    """Variational block inspired by the photonic fraud circuit, implemented with qubit gates."""

    def __init__(self, params: FraudLayerParameters):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        # Parameterised single‑qubit gates
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)
        # Additional gates to emulate photonic operations
        self.h = tq.H()
        self.cnot = tq.CNOT()

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=3)
        self.crx(qdev, wires=[0, 2])
        self.h(qdev, wires=3)
        self.cnot(qdev, wires=[3, 0])


class HybridNATQuantumModel(tq.QuantumModule):
    """Quantum counterpart of HybridNATModel."""
    def __init__(self, fraud_params: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layers = nn.ModuleList(
            FraudQuantumLayer(param) for param in fraud_params
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Reduce spatial information to a vector matching the encoder input size
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        for layer in self.q_layers:
            layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["HybridNATQuantumModel", "FraudLayerParameters", "FraudQuantumLayer"]
