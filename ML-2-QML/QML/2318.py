"""Quantum hybrid GraphQNN inspired by GraphQNN and Quantum‑NAT.

The `GraphQNNGen121Quantum` module encodes node features into a
quantum state via a tensor‑product of single‑qubit states, applies
a stack of random and parameterized quantum layers, then measures
all qubits.  The output is post‑processed by a classical linear
layer to match the 4‑dimensional classical counterpart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Sequence

from.GraphQNN import (
    fidelity_adjacency,
)

class GraphQNNGen121Quantum(tq.QuantumModule):
    """Quantum hybrid GNN + CNN model.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture of the underlying graph neural network.
    n_wires : int, default 4
        Number of qubits used for the quantum part.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT()

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.cnot(qdev, wires=[0, 1])

    def __init__(self, qnn_arch: Sequence[int], n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.fc = nn.Linear(n_wires, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Input images of shape (B, 1, 28, 28).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Encode image into qubits via average pooling
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.fc(out)
        return self.norm(out)

__all__ = ["GraphQNNGen121Quantum"]
