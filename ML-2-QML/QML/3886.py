"""QuantumHybridRegressor – quantum component of the hybrid regression pipeline.

Implements the variational circuit and expectation‑based read‑out used by the classical wrapper. The module is fully differentiable and can be integrated into any PyTorch training loop.
"""

from __future__ import annotations

import torch
import torchquantum as tq
import torch.nn as nn


class QuantumLayer(tq.QuantumModule):
    """Variational block: random layer followed by trainable RX/RY on each wire."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)


class QuantumRegressor(tq.QuantumModule):
    """Quantum regression head: encodes the input state, applies the variational layer,
    measures Pauli‑Z on all wires, and passes the result through a linear layer."""

    def __init__(self, num_wires: int, shift: float = 0.0):
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.qlayer = QuantumLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.shift = shift

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.qlayer.n_wires,
                                bsz=bsz,
                                device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.qlayer(qdev)
        features = self.measure(qdev)
        return (self.head(features) + self.shift).squeeze(-1)


__all__ = ["QuantumLayer", "QuantumRegressor"]
