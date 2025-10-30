"""
Quantum sampler with a two‑layer variational circuit.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np


class SamplerQNNGen180:
    """
    Variational sampler implemented with Pennylane.

    Circuit layout:
        - Layer 1: RY rotations on each qubit (parameterized by `weights1`)
        - Entangling block: CNOTs in a ring
        - Layer 2: RY rotations on each qubit (parameterized by `weights2`)
        - Measurement: sample bitstrings using the built‑in sampler
    """

    def __init__(self, num_qubits: int = 2) -> None:
        self.num_qubits = num_qubits
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.weights1 = np.random.uniform(0, 2 * np.pi, (num_qubits,))
        self.weights2 = np.random.uniform(0, 2 * np.pi, (num_qubits,))

        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights1, weights2):
            for i in range(num_qubits):
                qml.RY(weights1[i], wires=i)
            # Ring entanglement
            for i in range(num_qubits):
                qml.CNOT(wires=[i, (i + 1) % num_qubits])
            for i in range(num_qubits):
                qml.RY(weights2[i], wires=i)
            return qml.sample(qml.PauliZ(wires=range(num_qubits)))

        self.circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Generate a probability distribution over 2^num_qubits outcomes.

        Parameters
        ----------
        inputs : np.ndarray
            Input parameters that modulate the first layer rotations.
            Shape must be (num_qubits,).

        Returns
        -------
        np.ndarray
            Normalized probability vector of length 2^num_qubits.
        """
        # Combine external inputs with internal weights
        weights1 = inputs + self.weights1
        weights2 = self.weights2
        samples = self.circuit(weights1, weights2)
        # Convert samples to bitstring indices
        bitstrings = np.array([[int(bit) for bit in s[::-1]] for s in samples])
        indices = bitstrings.dot(1 << np.arange(self.num_qubits))
        counts = np.bincount(indices, minlength=2 ** self.num_qubits)
        probs = counts / counts.sum()
        return probs


__all__ = ["SamplerQNNGen180"]
