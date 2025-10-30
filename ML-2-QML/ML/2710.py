import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, List, Sequence
# Import quantum components from the QML module
# The QML module is expected to be in the same package and named qml_hybrid
from.qml_hybrid import QuantumCircuitWrapper, QuantumHybridLayer, FastHybridEstimator
from qiskit.providers.aer import AerSimulator

class HybridQCNet(nn.Module):
    """Classical‑quantum binary classifier with CNN backbone and differentiable quantum head.

    This module merges the CNN architecture from the original ClassicalQuantumBinaryClassification
    seed with a parameterised quantum circuit and a FastEstimator for efficient evaluation.
    The quantum head is wrapped in a custom autograd function that implements the parameter
    shift rule, enabling end‑to‑end training with PyTorch optimizers.
    """

    def __init__(
        self,
        backend=None,
        shots: int = 100,
        shift: float = np.pi / 2,
        n_qubits: int = 2,
    ) -> None:
        super().__init__()
        # Convolutional backbone identical to the original seed
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Quantum hybrid layer
        self.quantum_circuit = QuantumCircuitWrapper(
            n_qubits=n_qubits,
            backend=backend or AerSimulator(),
            shots=shots,
        )
        self.shift = shift
        self.quantum_layer = QuantumHybridLayer.apply
        # Estimator for efficient batch evaluation
        self.estimator = FastHybridEstimator(
            circuit=self.quantum_circuit,
            shift=self.shift,
            shots=shots,
            noise=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN and quantum hybrid head."""
        x = F.relu(self.conv1(inputs))
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
        # Quantum hybrid layer expects a 1D tensor of shape (batch,)
        logits = x.squeeze(-1)
        probs = self.quantum_layer(logits, self.quantum_circuit, self.shift)
        probs = probs.unsqueeze(-1)
        return torch.cat((probs, 1 - probs), dim=-1)

    def evaluate_quantum_head(
        self,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Evaluate the quantum head on a batch of parameters using the FastHybridEstimator."""
        # Default observable: Z on first qubit
        from qiskit.quantum_info.operators import Pauli
        observable = Pauli("ZI")
        return self.estimator.evaluate([observable], parameter_sets)

__all__ = ["HybridQCNet"]
