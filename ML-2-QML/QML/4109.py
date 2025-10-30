"""Quantum implementation of the QuantumNATHybrid model.

The quantum module replaces the classical CNN backbone with a variational
circuit that emulates a quanvolution layer.  It inherits from
torchquantum.QuantumModule and mirrors the classical architecture
through residual‑like random layers and a final measurement head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F


class QResidualLayer(tq.QuantumModule):
    """Quantum analogue of a residual block using random gates."""

    def __init__(self, n_wires: int, n_ops: int = 30) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.random = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        # Small trainable rotation per wire
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Residual connection via a second random layer
        self.random(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)


class QuantumNATHybrid(tq.QuantumModule):
    """Variational circuit that mirrors the classical QuantumNATHybrid."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4

        # Encoder that maps classical features to qubit states
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )

        # Quanvolution‑like residual block
        self.q_residual = QResidualLayer(self.n_wires, n_ops=40)

        # Final variational layer
        self.final_layer = tq.RandomLayer(
            n_ops=50, wires=list(range(self.n_wires))
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical head
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )

        # 1. Encode classical input into qubits
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        self.encoder(qdev, pooled)

        # 2. Quanvolution‑like residual block
        self.q_residual(qdev)

        # 3. Final variational layer
        self.final_layer(qdev)

        # 4. Measurement
        out = self.measure(qdev)

        return self.norm(out)


__all__ = ["QuantumNATHybrid"]
