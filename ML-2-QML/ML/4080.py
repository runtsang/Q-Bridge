"""Hybrid quantum-classical binary classification network.

This module defines a PyTorch neural network that integrates 2‑D convolutions,
a quanvolutional quantum filter, a dense head, and a hybrid quantum
expectation or sampler head.  The network can be instantiated with either
a Qiskit‑based expectation circuit or a TorchQuantum sampler for
research flexibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np
import qiskit

# Import the quantum circuits defined in the QML module
from quantum_circuits import QuantumExpectationCircuit, SamplerQNNCircuit


class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
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
        return torch.cat(patches, dim=1)


class SamplerQNN(tq.QuantumModule):
    """Quantum sampler module derived from a 2‑qubit parameterised circuit."""
    def __init__(self):
        super().__init__()
        self.circuit = SamplerQNNCircuit()
        # learnable weights for the sampler circuit
        self.register_buffer("weights", torch.randn(4, requires_grad=True))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape (batch, 2)
        batch = inputs.shape[0]
        probs = []
        for i in range(batch):
            prob = self.circuit.run(
                inputs[i].tolist(), self.weights.detach().numpy()
            )
            probs.append(prob)
        return torch.tensor(probs, device=inputs.device)


class HybridExpectation(nn.Module):
    """Differentiable layer that feeds activations into a Qiskit expectation circuit."""
    def __init__(self, backend, shots: int = 200):
        super().__init__()
        self.circuit = QuantumExpectationCircuit()
        self.backend = backend
        self.shots = shots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, 1)
        x = x.squeeze()
        expectations = self.circuit.run(
            x.tolist(), backend=self.backend, shots=self.shots
        )
        return torch.tensor(expectations, device=x.device).unsqueeze(-1)


class QCNet(nn.Module):
    """Hybrid convolutional network for binary classification with multiple quantum modules."""
    def __init__(self, use_sampler_head: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        backend = qiskit.Aer.get_backend("aer_simulator")
        if use_sampler_head:
            self.head = SamplerQNN()
        else:
            self.head = HybridExpectation(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        logits = self.head(x)

        # If the head is a sampler we already have probabilities
        if isinstance(self.head, SamplerQNN):
            probs = logits
        else:
            probs = torch.sigmoid(logits)

        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuanvolutionFilter", "SamplerQNN", "HybridExpectation", "QCNet"]
