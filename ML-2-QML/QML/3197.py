"""Hybrid quantum fraud detection circuit combining photonic‑style parameterisation and a Quantum‑NAT inspired layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F


class PhotonicQuantumLayer(tq.QuantumModule):
    """Variational layer that emulates photonic operations with discrete‑qubit gates."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.random = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)


class QFCQuantumLayer(tq.QuantumModule):
    """Quantum‑NAT style layer operating on 4 qubits."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.random = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)


class FraudDetectionHybridQ(tq.QuantumModule):
    """Full hybrid quantum fraud detection model."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that maps a pooled image to qubit amplitudes
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.photonic_layer = PhotonicQuantumLayer()
        self.fcin_layer = QFCQuantumLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Reduce image to a 4‑dim vector via average pooling (mirrors CNN pooling)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Quantum device for the batch
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.photonic_layer(qdev)
        self.fcin_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["FraudDetectionHybridQ"]
