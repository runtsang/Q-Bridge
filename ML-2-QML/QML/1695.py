"""
SamplerQNNAdvanced – Quantum implementation
===========================================

This module implements a depth‑2 variational sampler using PennyLane.
The circuit operates on two qubits, applies parameterized rotations
and CNOT entangling gates, and returns a probability distribution
over the computational basis via a sampling backend.

Usage
-----
>>> from SamplerQNNAdvanced import SamplerQNNAdvanced
>>> sampler = SamplerQNNAdvanced()
>>> probs = sampler.sample([0.5, -0.3])
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class SamplerQNNAdvanced:
    """
    A parameterized quantum sampler with two qubits.

    The circuit consists of:
    - Two layers of RX/RZ rotations on each qubit (input parameters).
    - Two entangling CNOT layers (controlled‑NOT).
    - Two layers of RX/RZ rotations (trainable weights).

    The sampler returns the probability of measuring |00> and |01>
    (i.e., the first two basis states) as a 2‑dimensional vector.
    """

    def __init__(self, device_name: str = "default.qubit", wires: int = 2) -> None:
        self.dev = qml.device(device_name, wires=wires)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Input‑parameterized rotations
            qml.RX(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # Entangling layer 1
            qml.CNOT(wires=[0, 1])

            # Trainable rotations
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=1)

            # Entangling layer 2
            qml.CNOT(wires=[0, 1])

            # Trainable rotations
            qml.RX(weights[2], wires=0)
            qml.RY(weights[3], wires=1)

            # Sample from the computational basis
            return qml.sample()

        self.circuit = circuit
        # Initialize parameters
        self.inputs = pnp.array([0.0, 0.0], requires_grad=False)
        self.weights = pnp.random.uniform(-np.pi, np.pi, 4)

    def set_inputs(self, inputs: np.ndarray) -> None:
        """
        Update the input parameters of the circuit.

        Parameters
        ----------
        inputs : array‑like
            Two‑element array of rotation angles for the input layer.
        """
        self.inputs = pnp.array(inputs, requires_grad=False)

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Update the trainable weights of the circuit.

        Parameters
        ----------
        weights : array‑like
            Four‑element array of rotation angles for the trainable layers.
        """
        self.weights = pnp.array(weights, requires_grad=True)

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the variational circuit and return a 2‑dimensional probability
        distribution over the first two computational basis states.

        Parameters
        ----------
        inputs : array‑like
            Two‑element array of input rotation angles.

        Returns
        -------
        numpy.ndarray
            Probabilities for |00> and |01>.
        """
        self.set_inputs(inputs)
        samples = self.circuit(self.inputs, self.weights)

        # Count occurrences of each basis state
        counts = np.zeros(4, dtype=int)
        for s in samples:
            idx = int("".join(str(bit) for bit in s[::-1]), 2)
            counts[idx] += 1

        probs = counts / len(samples)
        return probs[:2]

    def train(self, data: np.ndarray, targets: np.ndarray, epochs: int = 200, lr: float = 0.01) -> None:
        """
        Simple gradient‑descent training loop for the circuit parameters.

        Parameters
        ----------
        data : array‑like
            Batch of input rotations (shape: n_samples, 2).
        targets : array‑like
            Desired probability distributions (shape: n_samples, 2).
        epochs : int, default 200
            Number of training iterations.
        lr : float, default 0.01
            Learning rate for the Adam optimiser.
        """
        opt = qml.AdamOptimizer(stepsize=lr)
        self.weights = pnp.random.uniform(-np.pi, np.pi, 4)

        for _ in range(epochs):
            def loss(weights):
                probs_batch = np.array(
                    [self.circuit(x, weights)[0] for x in data]
                )
                return np.mean(
                    np.sum((probs_batch - targets) ** 2, axis=1)
                )

            self.weights = opt.step(loss, self.weights)

__all__ = ["SamplerQNNAdvanced"]
