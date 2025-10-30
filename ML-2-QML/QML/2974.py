"""Hybrid quantum model combining encoder, variational self‑attention, and measurement."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F

class QuantumSelfAttention(tq.QuantumModule):
    """Variational self‑attention block with per‑wire rotations and controlled‑rotations."""

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        for i in range(self.n_wires):
            self.rx(qdev, wires=i)
            self.ry(qdev, wires=i)
            self.rz(qdev, wires=i)
        for i in range(self.n_wires - 1):
            self.crx(qdev, wires=[i, i + 1])

class HybridNATModel(tq.QuantumModule):
    """Quantum encoder + self‑attention + measurement."""

    def __init__(self, n_wires: int = 4, num_classes: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.attn = QuantumSelfAttention(n_wires=n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, self.n_wires)
        self.encoder(qdev, pooled)
        self.attn(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridNATModel"]
