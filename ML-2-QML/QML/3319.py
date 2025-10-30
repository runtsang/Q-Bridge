"""Quantum regression model with QCNN‑style convolution and pooling layers.

The model encodes the classical feature vector into a quantum state,
processes it with a stack of parameterised convolution blocks, and finally
measures the qubits to obtain a feature vector that is fed to a classical
linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import encoder_op_list_name_dict


class QuantumConvolutionLayer(tq.QuantumModule):
    """QCNN inspired convolution block that applies a small parametric circuit
    to each adjacent pair of qubits.
    """

    def __init__(self, num_wires: int, n_ops: int = 10):
        super().__init__()
        self.num_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.num_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)


class HybridRegressionModel(tq.QuantumModule):
    """Full quantum regression model.

    The architecture follows a QCNN style:
    * Encoder: general feature map that maps the classical feature vector
      into a superposition over the qubit register.
    * Convolution blocks: repeated application of QuantumConvolutionLayer.
    * Head: linear layer mapping the quantum feature vector to a scalar regression target.
    """

    def __init__(self, num_wires: int, conv_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires

        # Encoder that uses a simple Ry rotation per wire
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict[f"{num_wires}xRy"])

        # Stack of convolution blocks
        self.blocks = nn.ModuleList([QuantumConvolutionLayer(num_wires) for _ in range(conv_layers)])

        # Measurement of the remaining qubits
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)

        # Encode classical features
        self.encoder(qdev, state_batch)

        # Apply convolution blocks
        for block in self.blocks:
            block(qdev)

        # Measure remaining qubits
        features = self.measure(qdev)

        # Classical head
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel"]
