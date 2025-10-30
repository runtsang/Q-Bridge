"""QuantumHybridRegression – quantum implementation.

This module implements a variational quantum module that can be
plugged into the classical wrapper defined in `QuantumHybridRegression.py`.

The design mirrors the encoder and measurement strategy of the
original quantum seed, while exposing a clean `forward` interface
compatible with the classical wrapper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq

__all__ = ["QuantumRegressionModule"]

class QuantumRegressionModule(tq.QuantumModule):
    """Variational quantum circuit for regression.

    The circuit encodes the input state using a parameter‑free
    GeneralEncoder (Ry on each wire) and then applies a RandomLayer
    followed by trainable RX/RY gates.  The expectation of Pauli‑Z
    on all wires is measured and fed into a classical linear head.
    """
    class QLayer(tq.QuantumModule):
        """Inner variational layer."""
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Use a Ry‑only encoder to match the classical feature map
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Encode, variational layer, measurement, and classical head."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)
