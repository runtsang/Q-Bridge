"""Quantum version of the hybrid Quantum‑NAT model using torchquantum.

This module implements a quantum circuit that replaces the classical
parameterized MLP from the ML version. The architecture mirrors the
classical model: a CNN encoder, a variational quantum layer, and
a linear head that maps the quantum measurement to the 4‑dimensional
output. The quantum layer is a small variational circuit with
random, RX, RY, RZ, CRX, Hadamard, SX, and CNOT gates.

The model is fully compatible with the classical version: both
produce a tensor of shape (batch, 4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class _QuantumLayer(tq.QuantumModule):
    """Variational circuit that operates on 4 qubits."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=80, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)
        self.hadamard = tq.Hadamard()
        self.sx = tq.SX()
        self.cnot = tq.CNot()

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=2)
        self.crx(qdev, wires=[0, 3])
        self.hadamard(qdev, wires=3)
        self.sx(qdev, wires=2)
        self.cnot(qdev, wires=[3, 0])

class QuantumNAT__gen503(tq.QuantumModule):
    """Quantum implementation of the hybrid Quantum‑NAT model."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = _QuantumLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(self.n_wires, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.head(out)
        return self.norm(out)

__all__ = ["QuantumNAT__gen503"]
