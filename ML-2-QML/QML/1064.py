"""SamplerQNN: Quantum sampler implemented with Pennylane.

This implementation builds a parameterized quantum circuit with two qubits,
entanglement via CNOT gates, and rotation layers that depend on both input
features and trainable weights. The circuit returns a probability distribution
over the computational basis states. The class exposes a `sample` method
that draws samples from the distribution using a quantum simulator.
"""

import pennylane as qml
import torch
from typing import Tuple

class SamplerQNN:
    """Quantum sampler network using Pennylane."""
    def __init__(self, dev: qml.Device | None = None, num_qubits: int = 2) -> None:
        self.num_qubits = num_qubits
        self.dev = dev or qml.device("default.qubit", wires=num_qubits)
        self.weight_shape = (4,)
        self.weights = torch.nn.Parameter(torch.randn(self.weight_shape))
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        self.sample_qnode = qml.QNode(self._sample_circuit, self.dev, interface="torch")

    def _circuit(self, inputs: torch.Tensor, weights: torch.Tensor) -> Tuple[float, float]:
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[2], wires=0)
        qml.RY(weights[3], wires=1)
        probs = qml.probs(wires=[0, 1])
        return probs[0], probs[1]

    def _sample_circuit(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[2], wires=0)
        qml.RY(weights[3], wires=1)
        return qml.sample(wires=range(self.num_qubits))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probs = self.qnode(inputs, self.weights)
        return torch.tensor(probs, dtype=torch.float32)

    def sample(self, num_samples: int = 1000) -> torch.Tensor:
        samples = self.sample_qnode(torch.zeros(2), self.weights, num_samples=num_samples)
        idx = torch.argmax(samples, dim=-1)
        return idx

__all__ = ["SamplerQNN"]
