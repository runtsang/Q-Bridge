"""Hybrid quantum model mirroring the classical counterpart.

The model uses a 4‑wire quantum device.  A general encoder maps the
spatially pooled input to the quantum state, followed by a parameterised
quantum layer that mixes the wires with a random circuit and trainable
RX/RY/RZ/CRX gates.  After measurement the resulting expectation values
are fed into a small classical MLP to produce the final four‑dimensional
output.  Batch‑norm is applied for stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQuantumNATModel(tq.QuantumModule):
    class QuantumEncoder(tq.QuantumModule):
        """General encoder using a fixed 4‑wire circuit."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict["4x4_ryzxy"]
            )

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.encoder(qdev)

    class QuantumLayer(tq.QuantumModule):
        """Parameterized quantum layer with random and trainable gates."""

        def __init__(self) -> None:
            super().__init__()
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(4))
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = self.QuantumEncoder()
        self.q_layer = self.QuantumLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.classical_fc = nn.Sequential(
            nn.Linear(self.n_wires, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out_q = self.measure(qdev)
        out = self.classical_fc(out_q)
        return self.norm(out)


__all__ = ["HybridQuantumNATModel"]
