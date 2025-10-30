"""Quantum sampler built with PennyLane, supporting variational sampling and hybrid training."""
from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Sequence, List, Tuple


class SamplerQNN:
    """
    A parameterised quantum sampler implemented as a PennyLane QNode.
    Supports arbitrary input dimensions and multiple entangling layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (must match input dimensionality).
    num_layers : int, optional
        Number of variational layers. Defaults to 2.
    entanglement : str or Sequence[Tuple[int, int]], optional
        Entanglement pattern passed to PennyLane's variational circuit. Defaults to 'circular'.
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 2,
        entanglement: str | Sequence[Tuple[int, int]] = "circular",
    ) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.entanglement = entanglement
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        # Initialise trainable weights
        self.weights = np.random.randn(self.num_layers, self.num_qubits)

        # Build the variational circuit
        @qml.qnode(self.dev, interface="autograd", diff_method="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> List[float]:
            # Encode inputs via RY rotations
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            for layer in range(self.num_layers):
                for qubit in range(self.num_qubits):
                    qml.RY(weights[layer, qubit], wires=qubit)
                # Entangling layer
                qml.CNOT(wires=self.entanglement if isinstance(self.entanglement, int)
                         else (self.entanglement[0], self.entanglement[1]))

            return qml.probs(wires=range(self.num_qubits))

        self.circuit = circuit

    def sample(self, inputs: np.ndarray, num_samples: int = 1000) -> np.ndarray:
        """
        Generate samples from the quantum circuit by measuring in the computational basis.

        Parameters
        ----------
        inputs : np.ndarray
            Input parameters of shape (num_qubits,).
        num_samples : int, optional
            Number of samples to draw. Defaults to 1000.

        Returns
        -------
        np.ndarray
            Array of samples of shape (num_samples, num_qubits).
        """
        probs = self.circuit(inputs, self.weights)
        return np.random.choice(2, size=(num_samples, self.num_qubits), p=probs)

    def loss_and_grad(self, inputs: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute negative log‑likelihood loss and its gradient for hybrid training.

        Parameters
        ----------
        inputs : np.ndarray
            Input parameters.
        target : np.ndarray
            One‑hot target distribution of shape (2**num_qubits,).

        Returns
        -------
        Tuple[float, np.ndarray]
            Loss value and gradient w.r.t. trainable weights.
        """
        probs = self.circuit(inputs, self.weights)
        loss = -np.sum(target * np.log(probs + 1e-12))
        grads = qml.grad(self.circuit, argnum=1)(inputs, self.weights)
        return loss, grads

    def update_weights(self, grads: np.ndarray, lr: float = 0.01) -> None:
        """
        Gradient descent update of trainable weights.

        Parameters
        ----------
        grads : np.ndarray
            Gradient array matching self.weights shape.
        lr : float, optional
            Learning rate. Defaults to 0.01.
        """
        self.weights -= lr * grads


__all__ = ["SamplerQNN"]
