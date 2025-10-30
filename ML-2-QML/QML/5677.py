"""Quantum regression head used in the hybrid fraud‑detection / regression model.

The module defines a variational circuit that encodes a 2‑dimensional classical state
into a quantum state of `num_wires` qubits, applies a trainable random layer
followed by single‑qubit rotations, measures in the Pauli‑Z basis, and maps the
measurement outcomes to a scalar regression output via a linear layer.
"""

import torch
import torch.nn as nn
import torchquantum as tq

class QuantumRegressionHead(tq.QuantumModule):
    """Variational quantum regression head."""
    def __init__(self, num_wires: int):
        super().__init__()
        # Encoder: a fixed 2‑qubit Ry circuit for each wire
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Variational part: random layer + two trainable rotations per wire
        self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        # Measurement: Pauli‑Z on all wires
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head: map measurement vector to scalar
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Encode, process, and regress."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        for wire in range(self.q_layer.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        features = self.measure(qdev)  # shape (batch, num_wires)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegressionHead"]
