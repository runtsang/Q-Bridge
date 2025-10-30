"""
Variational Quantum Circuit for Quantum‑NAT with parameter‑shift training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class VariationalLayer(tq.QuantumModule):
    """Parameterized variational layer with entangling gates."""

    def __init__(self, n_wires: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Parameterized single‑qubit rotations
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        # Entangling CNOTs
        self.cnot = tq.CNOT

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        for _ in range(self.n_layers):
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            for w in range(self.n_wires - 1):
                self.cnot(qdev, wires=[w, w + 1])
            self.cnot(qdev, wires=[self.n_wires - 1, 0])


class QuantumNatModel(tq.QuantumModule):
    """Hybrid quantum model with amplitude‑encoding and variational layer."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder: amplitude encoding via Ry/Rz rotations
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.variational = VariationalLayer(n_wires=self.n_wires, n_layers=4)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Reduce 28x28 image to 16 features
        pooled = torch.nn.functional.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.variational(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QuantumNatModel"]
