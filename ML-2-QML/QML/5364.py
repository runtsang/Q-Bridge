"""Hybrid regression quantum model combining feature mapping, variational ansatz, and measurement head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq

class QuantumSelfAttention(tq.QuantumModule):
    """Quantum circuit performing a self‑attention style operation."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_qubits}xRy"])
        self.attention_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        self.attention_layer(qdev)
        return self.measure(qdev)

class QConvLayer(tq.QuantumModule):
    """Convolution‑style variational ansatz on paired qubits."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.cnot = tq.CNOT()
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        for i in range(0, self.n_wires, 2):
            self.rx(qdev, wires=i)
            self.ry(qdev, wires=i+1)
            self.cnot(qdev, wires=[i, i+1])

class HybridRegressionModel(tq.QuantumModule):
    """Hybrid quantum regression model combining attention, convolution, and pooling."""
    def __init__(self, num_wires: int = 16):
        super().__init__()
        self.n_wires = num_wires
        self.attention = QuantumSelfAttention(num_wires)
        self.conv1 = QConvLayer(num_wires)
        self.pool1 = QConvLayer(num_wires // 2)  # simplified pooling by halving wires
        self.head = nn.Linear(num_wires // 2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode classical data into qubits
        self.attention.encoder(qdev, state_batch)
        # Apply attention circuit
        _ = self.attention(qdev)
        # Apply convolution layer
        self.conv1(qdev)
        # Pooling
        self.pool1(qdev)
        # Measure final state
        features = self.attention.measure(qdev)
        pooled = features[:, :self.n_wires // 2]
        return self.head(pooled).squeeze(-1)

__all__ = ["HybridRegressionModel"]
