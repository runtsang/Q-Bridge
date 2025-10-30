"""Quantum hybrid binary classifier using Qiskit EstimatorQNN.

The network consists of a classical CNN backbone that produces two parameters
(θ, φ). These parameters are fed into a 1‑qubit EstimatorQNN circuit
(H‑Ry‑Rx) whose expectation of the Pauli‑Y observable is computed via a
StatevectorEstimator. The expectation is passed through a sigmoid to obtain
class probabilities.  The implementation uses Qiskit and the qiskit‑machine‑learning
package.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class HybridBinaryClassifier(nn.Module):
    """CNN + EstimatorQNN head for binary classification."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        # Classical backbone identical to the ML version
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(540, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # outputs [theta, phi]
        self.shift = shift

        # Build the 1‑qubit EstimatorQNN circuit
        theta = Parameter("θ")
        phi = Parameter("φ")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(theta, 0)
        qc.rx(phi, 0)
        # Observable Y
        observable = [("Y", 1)]
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[theta, phi],
            weight_params=[],
            estimator=estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        params = self.fc3(x)  # (batch, 2)
        theta = params[..., 0]
        phi = params[..., 1]
        expectation = self.estimator_qnn(torch.stack([theta, phi], dim=1))
        logits = expectation + self.shift
        probs = torch.sigmoid(logits)
        return torch.stack([probs, 1 - probs], dim=-1)
