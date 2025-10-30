from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from typing import List

class HybridConvRegressor(tq.QuantumModule):
    """
    Quantum hybrid model that encodes a 2‑D convolutional feature vector into
    qubits, applies a variational circuit inspired by fraud‑detection
    photonic layers, and outputs a regression score.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, n_wires: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.threshold = threshold
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRZ"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: Tensor of shape (batch, 2) representing the flattened
        convolutional patch after activation.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        # Threshold gating on measurement outcomes
        features = torch.where(features > self.threshold, torch.ones_like(features), torch.zeros_like(features))
        return self.head(features).squeeze(-1)

__all__ = ["HybridConvRegressor"]
