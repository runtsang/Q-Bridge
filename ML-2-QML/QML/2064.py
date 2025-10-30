"""Quantum sampler using a variational circuit with entanglement.

The SamplerQNN class builds a 2‑qubit variational circuit with three
parameterized layers of RY rotations and CZ entanglement. The circuit
is wrapped in a Pennylane QNode that returns the probability of
measuring the computational basis states. A simple sampling routine
is also provided.

This implementation is fully compatible with the classical SamplerQNN
above and can be used in hybrid training loops.
"""

from __future__ import annotations

import pennylane as qml
import torch
from pennylane import numpy as np


class SamplerQNN:
    """Quantum sampler network.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default 2).
    layers : int
        Number of variational layers (default 3).
    device_name : str
        Device for simulation, e.g., 'default.qubit' (default).
    """

    def __init__(
        self,
        n_qubits: int = 2,
        layers: int = 3,
        device_name: str = "default.qubit",
    ) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device(device_name, wires=n_qubits)

        # Parameter shape: (layers, n_qubits)
        self.params_shape = (layers, n_qubits)

        # Build the QNode
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def circuit(params: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
            """Variational circuit that accepts external inputs and trainable
            parameters and returns the full probability distribution over
            the 2^n_qubits computational basis states."""
            # Encode classical inputs as RY rotations
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            for l in range(self.layers):
                # RY rotations
                for i in range(self.n_qubits):
                    qml.RY(params[l, i], wires=i)
                # Entanglement: CZ between consecutive qubits
                for i in range(self.n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            # Measure probabilities
            return qml.probs(wires=range(self.n_qubits))

        self.circuit = circuit

    def forward(
        self,
        inputs: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the circuit and return a probability distribution."""
        return self.circuit(params, inputs)

    def sample(self, inputs: torch.Tensor, params: torch.Tensor, n_samples: int = 1000):
        """Return samples drawn from the circuit."""
        probs = self.forward(inputs, params).detach().numpy().flatten()
        # Sample indices
        samples = np.random.choice(
            2 ** self.n_qubits, size=n_samples, p=probs
        )
        # Convert indices to bit strings
        bitstrings = [
            format(idx, f"0{self.n_qubits}b") for idx in samples
        ]
        return bitstrings


__all__ = ["SamplerQNN"]
