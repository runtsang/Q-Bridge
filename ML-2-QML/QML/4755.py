"""Quantum estimator using PennyLane.

The module defines `UnifiedEstimatorQNN`, a `pennylane.nn.PennylaneModule` that maps
classical input features to a quantum state, runs a variational circuit, and
returns a vector of expectation values.  The number of qubits equals the number
of features, ensuring maximal expressivity.  The circuit uses an
`AngleEmbedding` for the input and a stack of `BasicEntanglerLayers` as the
variational ansatz.  The returned vector can be fed into a classical head
for regression or classification.

Typical usage:

>>> from quantum_estimator import UnifiedEstimatorQNN
>>> qnn = UnifiedEstimatorQNN(num_features=3, quantum_depth=2)
>>> x = torch.tensor([[0.1, 0.5, -0.2]], dtype=torch.float32)
>>> out = qnn(x)   # shape: (1, 3)
"""

import pennylane as qml
import pennylane.numpy as np
import torch


class UnifiedEstimatorQNN(qml.nn.PennylaneModule):
    """Variational quantum estimator that outputs expectation values of PauliZ."""

    def __init__(self, num_features: int, quantum_depth: int = 2, device: str = "default.qubit"):
        super().__init__()
        self.num_features = num_features
        self.quantum_depth = quantum_depth

        # Create a quantum device
        self.dev = qml.device(device, wires=num_features)

        # Trainable variational parameters
        self.variational_weights = torch.randn(num_features, 3, requires_grad=True)

        # Define the variational circuit
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor):
            # Embed classical data
            qml.templates.AngleEmbedding(inputs, wires=range(num_features))
            # Variational layers
            for _ in range(quantum_depth):
                qml.templates.BasicEntanglerLayers(
                    weights=self.variational_weights, wires=range(num_features)
                )
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(num_features)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return expectation values for a batch of inputs."""
        batch_outputs = []
        for sample in x:
            batch_outputs.append(self.circuit(sample))
        return torch.stack(batch_outputs)


__all__ = ["UnifiedEstimatorQNN"]
