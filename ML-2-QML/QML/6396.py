from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import qiskit
from qiskit import Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class QuantumHybridLayer(nn.Module):
    """Parameter‑shift layer that forwards activations through a Qiskit EstimatorQNN."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        theta = Parameter("theta")
        qc = qiskit.QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        qc.barrier()
        qc.ry(theta, 0)
        qc.rx(theta, 1)
        self.circuit = qc

        self.obs = SparsePauliOp.from_list([("Y" + "I" * (n_qubits - 1), 1)])
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.obs,
            input_params=[theta],
            weight_params=[],
            estimator=self.estimator,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # EstimatorQNN expects a 1‑D input of shape (batch,)
        return self.estimator_qnn(inputs.squeeze()).unsqueeze(-1)

class HybridBinaryNet(nn.Module):
    """Hybrid CNN that replaces the final dense head with a quantum expectation value."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.quantum_head = QuantumHybridLayer()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        qout = self.quantum_head(x)
        probs = torch.sigmoid(qout)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryNet"]
