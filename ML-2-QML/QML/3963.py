"""Quantum regression model implementing a lightweight variational circuit.

The class `QModel` is a subclass of `torchquantum.QuantumModule`.  It encodes
classical input features onto a quantum state, applies a random layer and
trainable rotations, measures Pauli‑Z expectation values, and finally
regresses to a scalar output.  This module can be trained end‑to‑end with
classical optimizers and is fully differentiable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq

class QModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        """Inner layer that applies a random circuit followed by
        trainable single‑qubit rotations."""
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
        # Encoder: maps classical data onto a product of Ry gates
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Run the variational circuit and return a scalar regression output."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=state_batch.device
        )
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QModel"]
