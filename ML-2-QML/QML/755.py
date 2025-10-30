"""Quantum sampler network using a parameterized variational circuit.

This module defines a SamplerQNN class that wraps a Pennylane
QNode.  The circuit operates on two qubits, takes a two‑element
classical input, and uses trainable weight parameters to generate
a probability distribution over the computational basis states.
The architecture includes an entangling layer and a repeatable
parameterized block, making it suitable for variational training.
"""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from typing import Tuple


class SamplerQNN:
    """
    Quantum sampler network.

    Parameters
    ----------
    dev : pennylane.Device, optional
        Quantum device to use. Defaults to a local statevector
        simulator with 2 qubits.
    """

    def __init__(self, dev: qml.Device | None = None) -> None:
        self.dev = dev or qml.device("default.qubit", wires=2)

        # Number of trainable parameters per layer
        self.num_params_per_layer = 4

        # Build the QNode
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: Tuple[float, float], weights: np.ndarray):
            # Encode inputs
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # Entangling layer
            qml.CNOT(wires=[0, 1])

            # Parameterized block
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)

            # Measurement: probabilities of |00>, |01>, |10>, |11>
            probs = qml.probs(wires=[0, 1])
            return probs

        self.circuit = circuit

        # Initialize trainable weights
        self.weights = np.random.randn(self.num_params_per_layer)

    def forward(self, inputs: Tuple[float, float]) -> np.ndarray:
        """
        Evaluate the circuit and return the probability distribution.

        Parameters
        ----------
        inputs : tuple of float
            Classical input vector of length 2.

        Returns
        -------
        np.ndarray
            Probability vector of shape (4,) corresponding to
            the computational basis states.
        """
        return self.circuit(inputs, self.weights)

    def loss(self, inputs: Tuple[float, float], target: np.ndarray) -> float:
        """
        Simple cross‑entropy loss between the circuit output and a target distribution.

        Parameters
        ----------
        inputs : tuple of float
            Classical input vector.
        target : np.ndarray
            Target probability distribution of shape (4,).

        Returns
        -------
        float
            Cross‑entropy loss.
        """
        probs = self.forward(inputs)
        # Avoid log(0)
        eps = 1e-12
        return -np.sum(target * np.log(probs + eps))

    def grad(self, inputs: Tuple[float, float], target: np.ndarray) -> np.ndarray:
        """
        Compute gradients of the loss with respect to the trainable weights.

        Parameters
        ----------
        inputs : tuple of float
            Classical input vector.
        target : np.ndarray
            Target distribution.

        Returns
        -------
        np.ndarray
            Gradient vector of shape (num_params_per_layer,).
        """
        return qml.grad(self.loss)(inputs, target)

    def update_weights(self, grads: np.ndarray, lr: float = 0.01) -> None:
        """
        Update the trainable weights using a simple gradient descent step.

        Parameters
        ----------
        grads : np.ndarray
            Gradient vector.
        lr : float
            Learning rate.
        """
        self.weights -= lr * grads

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dev={self.dev.name}, weights={self.weights})"
