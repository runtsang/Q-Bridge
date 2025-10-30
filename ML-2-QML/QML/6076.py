import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class HybridQuantumBinaryClassifier(nn.Module):
    """
    Quantum‑augmented CNN for binary classification.
    The convolutional feature extractor is identical to the classical
    counterpart, but the final head is a variational quantum circuit
    implemented via Qiskit’s EstimatorQNN.  The circuit uses two
    parameters: an input‑dependent `theta_in` and a trainable weight
    `theta_w`.  The expectation value of the Y‑observable
    serves as the logit for the binary task.
    """
    def __init__(self, shots: int = 1024, backend=None):
        super().__init__()
        # Classical feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum layer
        backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta_in = Parameter("theta_in")
        self.theta_w = Parameter("theta_w")
        self.circuit.h(0)
        self.circuit.ry(self.theta_in, 0)
        self.circuit.rx(self.theta_w, 0)
        self.circuit.measure_all()

        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=SparsePauliOp.from_list([("Y", 1)]),
            input_params=[self.theta_in],
            weight_params=[self.theta_w],
            estimator=StatevectorEstimator(backend=backend, shots=shots)
        )

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
        # Pass through quantum layer
        logits = self.qnn(x.squeeze(-1))
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
