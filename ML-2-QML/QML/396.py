"""Quantum fully connected layer using Pennylane.

Features
--------
- Parameterised Ry/RZ layers with optional entanglement.
- Automatic gradient computation via Pennylane's autograd.
- Simple training helper that runs one gradient‑descent step.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Iterable, Sequence


class FCL:
    """
    Variational quantum circuit mimicking a fully connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits / parameters (one per qubit per layer).
    n_layers : int, optional
        Number of variational layers.  Each layer consists of a Ry+RZ on every qubit
        followed by a CNOT ladder for entanglement.
    device : str | qml.Device, optional
        Pennylane device.  Defaults to the CPU simulator.
    """

    def __init__(self,
                 n_qubits: int = 1,
                 n_layers: int = 1,
                 device: qml.Device | str | None = None) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device or "default.qubit", wires=n_qubits)

        # Trainable parameters: theta[l][q][gate]
        self.params_shape = (n_layers, n_qubits, 2)  # Ry, RZ
        self.params = np.random.uniform(0, 2 * np.pi, self.params_shape)

        # Construct the quantum node
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: np.ndarray) -> np.ndarray:
            for l in range(n_layers):
                for q in range(n_qubits):
                    qml.RY(params[l, q, 0], wires=q)
                    qml.RZ(params[l, q, 1], wires=q)
                # Entangle with a CNOT ladder
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            # Expectation of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit with a flattened parameter vector.

        Parameters
        ----------
        thetas : Iterable[float]
            Flattened parameters matching ``self.params_shape``.
        Returns
        -------
        np.ndarray
            Expectation value as a 1‑D array.
        """
        params = np.array(thetas).reshape(self.params_shape)
        return np.array([self.circuit(params)])

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def train_step(self,
                   thetas: Iterable[float],
                   target: float,
                   lr: float = 1e-3) -> float:
        """
        One gradient‑descent step using Pennylane's autograd.

        Parameters
        ----------
        thetas : Iterable[float]
            Current flattened parameters.
        target : float
            Desired output.
        lr : float, optional
            Learning rate.

        Returns
        -------
        float
            Loss value (MSE) after the update.
        """
        params = np.array(thetas).reshape(self.params_shape)

        def loss_fn(p):
            pred = self.circuit(p)
            return (pred - target) ** 2

        loss, grads = qml.grad(loss_fn)(params)
        # Flatten grads for update
        grads_flat = grads.reshape(-1)
        # Simple gradient descent
        params_flat = params.reshape(-1) - lr * grads_flat
        # Update internal parameters
        self.params = params_flat.reshape(self.params_shape)
        return float(loss)

__all__ = ["FCL"]
