"""Hybrid quantum model that uses a CNN encoder followed by a variational circuit, matching the classical architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Iterable, Tuple

class HybridNATModel(tq.QuantumModule):
    """Quantum model that mirrors the classical HybridNATModel."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, depth: int):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            # Parameterized single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for i in range(self.n_wires):
                self.rx(qdev, wires=i)
                self.ry(qdev, wires=i)
                self.rz(qdev, wires=i)
            # Entanglement pattern (CZ chain)
            for i in range(self.n_wires - 1):
                tqf.cz(qdev, wires=[i, i+1])
            # Depth‑repeated variational layer
            for _ in range(self.depth - 1):
                for i in range(self.n_wires):
                    self.crx(qdev, wires=[i, (i+1)%self.n_wires])

    def __init__(self, n_wires: int = 4, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires, depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridNATModel"]
