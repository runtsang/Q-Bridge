"""Quantum self‑attention block built with PennyLane.

The class implements a variational circuit that applies parameterized rotations
and entangling gates to a set of qubits.  The circuit can be trained with
PennyLane's automatic differentiation and run on either a simulator or a
real backend.  The `run` method returns measurement counts, while the `train`
method demonstrates how to optimize the rotation and entanglement parameters
using a simple loss function.
"""

import pennylane as qml
import numpy as np
from typing import Tuple

class SelfAttentionEnhanced:
    """Variational quantum self‑attention circuit."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Builds the variational circuit."""
        for i in range(self.n_qubits):
            qml.RX(rotation_params[3 * i], wires=i)
            qml.RY(rotation_params[3 * i + 1], wires=i)
            qml.RZ(rotation_params[3 * i + 2], wires=i)

        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])

        # Return measurement probabilities
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> dict:
        """
        Execute the circuit on a simulator and return measurement counts.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape (3 * n_qubits,) containing rotation angles.
        entangle_params : np.ndarray
            Array of shape (n_qubits - 1,) containing entangling angles.
        shots : int, optional
            Number of shots for sampling.  Defaults to 1024.

        Returns
        -------
        dict
            Dictionary of measurement outcome counts.
        """
        @qml.qnode(self.dev, interface="numpy", shots=shots)
        def qnode():
            self._circuit(rotation_params, entangle_params)
            return [qml.sample(qml.PauliZ(i)) for i in range(self.n_qubits)]

        samples = qnode()
        # Convert samples to bitstrings
        bitstrings = []
        for s in zip(*samples):
            bits = ''.join(['1' if bit == 1 else '0' for bit in s])
            bitstrings.append(bits)

        counts = {}
        for bits in bitstrings:
            counts[bits] = counts.get(bits, 0) + 1
        return counts

    def train(self,
              rotation_params: np.ndarray,
              entangle_params: np.ndarray,
              target: np.ndarray,
              lr: float = 0.01,
              epochs: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple training loop that optimizes the circuit parameters to match a
        target expectation value vector.

        Parameters
        ----------
        rotation_params : np.ndarray
            Initial rotation parameters.
        entangle_params : np.ndarray
            Initial entanglement parameters.
        target : np.ndarray
            Target vector of expectation values (length n_qubits).
        lr : float, optional
            Learning rate.  Defaults to 0.01.
        epochs : int, optional
            Number of training iterations.  Defaults to 10.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Final rotation and entanglement parameters.
        """
        rot = qml.numpy.array(rotation_params, requires_grad=True)
        ent = qml.numpy.array(entangle_params, requires_grad=True)

        opt = qml.GradientDescentOptimizer(stepsize=lr)

        for _ in range(epochs):
            def loss_fn():
                out = self._circuit(rot, ent)
                return qml.math.sum((out - target) ** 2)

            rot, ent = opt.step_and_cost(loss_fn, rot, ent)
        return rot.numpy(), ent.numpy()

__all__ = ["SelfAttentionEnhanced"]
