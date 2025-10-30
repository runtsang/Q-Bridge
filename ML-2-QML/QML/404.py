"""Quantum variant of QuantumNATEnhanced with a parameterised circuit and attention‑aware readout."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATEnhanced(tq.QuantumModule):
    """
    Quantum module extending the seed Quantum‑NAT.
    Enhancements:
        * Parameterised circuit block (`ParamCircuit`) with trainable rotations.
        * Attention‑aware measurement: each qubit output is weighted by a learnable scalar before normalisation.
    """
    class ParamCircuit(tq.QuantumModule):
        """A tunable circuit with layers of random rotations and entangling gates."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT()

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            for i in range(self.n_wires - 1):
                self.cnot(qdev, wires=[i, i + 1])

    def __init__(self, n_wires: int = 4, num_classes: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.circuit = self.ParamCircuit(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # learnable attention weights for each qubit
        self.attn_weights = nn.Parameter(torch.ones(n_wires))
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                 device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)  # match seed
        self.encoder(qdev, pooled)
        self.circuit(qdev)
        out = self.measure(qdev)  # (B, n_wires)
        # attention‑weighted readout
        out = out * self.attn_weights
        out = torch.sum(out, dim=1, keepdim=True)  # collapse to one value per batch
        out = self.norm(out)
        return out


__all__ = ["QuantumNATEnhanced"]
