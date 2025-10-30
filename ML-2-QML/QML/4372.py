"""Quantum implementation of QFCModel using TorchQuantum.

The quantum version mirrors the classical architecture but replaces the
classical encoder and quantum‑inspired layer with a real quantum circuit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum import encoder_op_list_name_dict


class QFCModel(tq.QuantumModule):
    """Quantum hybrid model that mirrors the classical QFCModel but uses a real quantum circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT()

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
            for w in range(0, self.n_wires - 1, 2):
                self.cnot(qdev, wires=[w, w + 1])

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder uses a simple Ry gate per wire
        self.encoder = tq.GeneralEncoder(
            encoder_op_list_name_dict[f"{n_wires}xRy"]
        )
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True
        )
        # Pooling similar to classical version
        pooled = F.avg_pool2d(x, 6).view(bsz, -1)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        features = self.measure(qdev)
        out = self.head(features)
        return self.norm(out)

    @staticmethod
    def kernel_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Quantum kernel evaluated via a fixed ansatz."""
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        qdev = tq.QuantumDevice(n_wires=4)
        for wire in range(4):
            tq.RY(qdev, wires=wire, params=a[:, wire] - b[:, wire])
        return torch.abs(qdev.states.view(-1)[0])


__all__ = ["QFCModel"]
