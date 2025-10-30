"""Quantum variant of the hybrid NAT model.  It processes the same
classical pooled features through a variational layer composed of a
RandomLayer, multiple parametric rotations, and a simple entangling
gate, then performs a Z‑measurement and a linear head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuantumNATGen(tq.QuantumModule):
    """Quantum model that ingests classical pooled features and
    applies a variational circuit before a linear readout."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Rich RandomLayer followed by trainable rotations
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Simple entanglement for expressivity
            self.cx = tq.CNOT(wires=[0, 1])

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            self.cx(qdev)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Classical encoder that maps 16‑dim pooled data to 4 qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4xRy"])
        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(self.n_wires, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical pooling to 16‑dim vector (matches the encoder)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        features = self.measure(qdev)
        out = self.head(features)
        return self.norm(out)


__all__ = ["QuantumNATGen"]
