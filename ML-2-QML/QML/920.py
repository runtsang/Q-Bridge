"""Quantum sampler network using Pennylane with variational entanglement and gradient‑based training."""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
from typing import Tuple

__all__ = ["SamplerQNN"]


class SamplerQNN:
    """
    Variational quantum sampler that maps a 2‑dimensional classical input to a probability distribution
    over 2 measurement outcomes. The circuit consists of input rotations, a configurable number of
    entangling layers, and trainable Ry rotations.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the device. Default is 2.
    num_layers : int
        Number of entangling layers. Default is 2.
    device_name : str
        Pennylane device name. Default is 'default.qubit'.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        num_layers: int = 2,
        device_name: str = "default.qubit",
    ) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device_name, wires=num_qubits)

        # Parameter placeholders for input and trainable weights
        self.input_params = pnp.array([0.0] * num_qubits, requires_grad=False)
        self.weight_params = pnp.array([0.0] * (num_qubits * num_layers), requires_grad=True)

        # Build the QNode
        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Input rotations
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)

            # Entangling layers with trainable Ry rotations
            idx = 0
            for _ in range(num_layers):
                # Apply CNOT chain for entanglement
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Apply Ry rotations with trainable weights
                for i in range(num_qubits):
                    qml.RY(weights[idx], wires=i)
                    idx += 1

            # Return measurement probabilities for |0> and |1> on each qubit
            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the probability distribution over measurement outcomes for a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (batch, num_qubits).

        Returns
        -------
        np.ndarray
            Probabilities of shape (batch, 2**num_qubits).
        """
        probs = []
        for x in inputs:
            probs.append(self.circuit(x, self.weight_params))
        return np.stack(probs)

    def sample(self, inputs: np.ndarray, num_shots: int = 1000) -> np.ndarray:
        """
        Sample bitstrings from the circuit for each input.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (batch, num_qubits).
        num_shots : int
            Number of shots per input.

        Returns
        -------
        np.ndarray
            Sampled bitstrings of shape (batch, num_shots, num_qubits).
        """
        samples = []
        for x in inputs:
            qml.set_options(shots=num_shots)
            samples.append(qml.sample(self.circuit, x, self.weight_params))
        return np.stack(samples)

    def train(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        loss_fn,
        optimizer: AdamOptimizer,
        epochs: int = 50,
    ) -> None:
        """
        Gradient‑based training loop for the variational sampler.

        Parameters
        ----------
        data : tuple
            Tuple of (inputs, targets) where targets are one‑hot encoded probability vectors.
        loss_fn : callable
            Loss function that accepts predictions and targets.
        optimizer : AdamOptimizer
            Pennylane optimizer.
        epochs : int
            Number of training epochs.
        """
        inputs, targets = data
        for epoch in range(epochs):
            preds = self.forward(inputs)
            loss = loss_fn(preds, targets)
            optimizer.step(lambda w: loss_fn(self.circuit(inputs, w), targets), self.weight_params)
            print(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f}")
