"""Hybrid quantum‑classical network that combines a quanvolutional quantum filter
with a Qiskit EstimatorQNN head.

The architecture mirrors the classical variant but replaces the final
classifier with a fully quantum neural network implemented via Qiskit’s
EstimatorQNN. A linear projection is used to reduce the 784‑dimensional
feature vector to a single scalar that serves as the input to the
parameterised quantum circuit. The expectation value of the circuit
is then linearly mapped to class logits.

This module is fully compatible with the original ``Quanvolution.py`` file
and can be used as a drop‑in replacement for the
``QuanvolutionClassifier`` class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class HybridQuanvolutionNet(nn.Module):
    """
    Quantum hybrid network that mimics the behaviour of a quanvolution filter
    followed by a Qiskit EstimatorQNN head.
    """
    def __init__(self, in_channels: int = 1, out_classes: int = 10) -> None:
        super().__init__()
        # Quantum filter parameters
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Linear projection from 784‑dimensional feature vector to a scalar
        self.projection = nn.Linear(4 * 14 * 14, 1)

        # Set up EstimatorQNN head
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(theta, 0)
        observable = SparsePauliOp.from_list([("Z", 1)])
        estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[theta],
            weight_params=[],
            estimator=estimator,
        )

        # Final linear layer to map expectation value to class logits
        self.classifier = nn.Linear(1, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Log‑softmax probabilities of shape (batch, out_classes).
        """
        bsz = x.shape[0]
        device = x.device
        # Prepare quantum device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        features = torch.cat(patches, dim=1)
        # Project features to a scalar
        projected = self.projection(features)  # shape: (batch, 1)
        # Evaluate the quantum neural network head
        qnn_output = self.qnn(projected)  # shape: (batch, 1)
        logits = self.classifier(qnn_output)  # shape: (batch, out_classes)
        return F.log_softmax(logits, dim=-1)

# Backward‑compatibility aliases
QuanvolutionFilter = HybridQuanvolutionNet
QuanvolutionClassifier = HybridQuanvolutionNet

__all__ = ["HybridQuanvolutionNet", "QuanvolutionFilter", "QuanvolutionClassifier"]
