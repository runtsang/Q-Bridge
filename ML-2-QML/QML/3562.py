"""
Hybrid quantum model: encoder, variational layer, and quantum‑kernel similarity to prototypes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict


class HybridNATModel(tq.QuantumModule):
    """Quantum encoder + variational layer + prototype‑based quantum kernel."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_prototypes: int = 4) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QLayer(self.n_wires)
        # Prototype states in the computational basis (complex parameters)
        init_state = torch.eye(2**self.n_wires, dtype=torch.cfloat)
        self.prototypes = nn.Parameter(init_state[:n_prototypes].clone())
        self.norm = nn.BatchNorm1d(n_prototypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        # Classical pooling to match encoder input dimensionality
        packed = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, packed)
        self.q_layer(qdev)
        # State vector after encoding and variational layer
        states = qdev.states.view(bsz, -1)
        # Quantum‑kernel similarity with prototypes
        proto = self.prototypes.unsqueeze(0)  # (1, P, 2**n)
        states_ = states.unsqueeze(1)  # (B, 1, 2**n)
        sim = torch.abs(torch.sum(states_ * proto.conj(), dim=2))  # (B, P)
        return self.norm(sim)


__all__ = ["HybridNATModel"]
