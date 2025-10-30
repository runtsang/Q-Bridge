"""SamplerQNN: Quantum neural sampler using PennyLane.

The circuit consists of two qubits, rotation gates parameterised by
input and trainable weights, entangling gates, and a measurement in
the computational basis.  The class exposes a `forward` method that
returns the probability distribution over the two‑bit basis states,
and a `sample` method that draws samples from this distribution.
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn

class SamplerQNN(nn.Module):
    """
    Quantum sampler implemented with PennyLane.
    """

    def __init__(self,
                 dev: str | qml.Device = "default.qubit",
                 num_qubits: int = 2,
                 weight_shape: tuple[int,...] = (4,)):
        super().__init__()
        self.dev = qml.device(dev, wires=num_qubits) if isinstance(dev, str) else dev
        self.num_qubits = num_qubits
        self.weight_shape = weight_shape

        # Initialise trainable weights
        self.weights = nn.Parameter(torch.randn(weight_shape))

        # Define the QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            # Input encoding: Ry rotations
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)

            # Entangling layer
            qml.CNOT(wires=[0, 1])

            # Parameterised rotations
            for i in range(num_qubits):
                qml.RY(weights[i], wires=i)

            # Second entangling layer
            qml.CNOT(wires=[0, 1])

            # Measurement: return probabilities of all basis states
            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return probability distribution over 2‑qubit basis states."""
        probs = self.circuit(inputs, self.weights)
        return probs

    def sample(self, inputs: torch.Tensor, num_samples: int) -> np.ndarray:
        """Draw samples from the quantum distribution."""
        probs = self.forward(inputs).detach().numpy()
        cum = np.cumsum(probs, axis=-1)
        rnd = np.random.rand(num_samples, probs.shape[0])
        return (rnd[..., None] < cum).sum(-1)

__all__ = ["SamplerQNN"]
