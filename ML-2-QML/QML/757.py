"""SamplerQNNExtended – a variational quantum sampler.

This implementation builds a multi‑layer entangling circuit with parameterised
single‑qubit rotations.  The sampler returns a probability distribution over
the two‑qubit computational basis by executing a state‑vector simulation
and sampling from the resulting state.  The circuit is differentiable via
parameter‑shift, enabling joint optimisation with a classical network.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Tuple


class SamplerQNNExtended:
    """
    Variational quantum sampler for two qubits.

    Circuit
    -------
    - 3 entangling layers.
    - Each layer contains:
        * Single‑qubit Ry rotations with trainable parameters.
        * CX gates forming a ring topology.
    - The circuit is parameterised by 2 * (3 * 2) = 12 rotation angles.
    - Sampling is performed using the state‑vector simulator.

    Methods
    -------
    sample(batch_size)
        Returns a probability distribution over the 2‑bit space.
    """

    def __init__(self, device: str = "default.qubit") -> None:
        self.dev = qml.device(device, wires=2, shots=None)
        self.params = np.random.uniform(0, 2 * np.pi, (3, 2), requires_grad=True)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: np.ndarray) -> np.ndarray:
            for layer in range(3):
                # Parameterised rotations
                qml.RY(params[layer, 0], wires=0)
                qml.RY(params[layer, 1], wires=1)
                # Entangling ring
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 0])
            return qml.state()

        self._circuit = circuit

    def sample(self, batch_size: int = 1) -> np.ndarray:
        """
        Sample from the quantum circuit's output distribution.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array of shape (batch_size, 2) with one‑hot encoded samples.
        """
        state = self._circuit(self.params)
        probs = np.abs(state) ** 2
        # Convert to probabilities over computational basis
        probs = probs.reshape(4)
        # Sample according to probabilities
        samples = np.random.choice(4, size=batch_size, p=probs)
        # One‑hot encode
        return np.eye(4)[samples][:, :2]

    def loss(self, target: np.ndarray) -> np.ndarray:
        """
        Cross‑entropy loss between the sampler's distribution and a target.

        Parameters
        ----------
        target : np.ndarray
            Target probability distribution of shape (2,).

        Returns
        -------
        np.ndarray
            Scalar loss value.
        """
        state = self._circuit(self.params)
        probs = np.abs(state) ** 2
        probs = probs.reshape(4)[:2]  # first two basis states
        return -np.sum(target * np.log(probs + 1e-12))

    def train_step(self, target: np.ndarray, lr: float = 0.01) -> None:
        """
        Perform a single gradient‑descent update on the circuit parameters.

        Parameters
        ----------
        target : np.ndarray
            Target probability distribution.
        lr : float
            Learning rate.
        """
        loss_val = self.loss(target)
        grads = qml.grad(self.loss)(self.params, target)
        self.params -= lr * grads


__all__ = ["SamplerQNNExtended"]
