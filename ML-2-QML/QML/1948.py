"""
Variational quantum fully connected layer using Pennylane.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class QuantumFullyConnectedLayer:
    """
    A lightweight variational circuit that emulates a fully‑connected layer.
    The circuit consists of alternating layers of RY rotations (parameterized
    by the input vector) and entangling CNOTs, followed by a measurement of Pauli‑Z.
    The output is the expectation value of the last qubit and can be differentiated
    automatically via Pennylane’s autograd interface.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be >= 1).
    dev : Optional[pennylane.Device]
        Quantum device to execute on.  Defaults to a 2‑qubit qasm simulator.
    cost_func : Optional[Callable]
        User‑supplied cost function for training; if None, the layer is
        evaluated in inference mode.

    Notes
    -----
    The ``run`` method accepts a vector of parameters matching the number of qubits
    and returns the expectation value of the last qubit.  Gradient computation
    is exposed through ``gradient`` and ``train_step`` helpers.
    """

    def __init__(self, n_qubits: int = 1, dev: qml.Device | None = None) -> None:
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits, shots=1024)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(thetas):
            for i in range(n_qubits):
                qml.RY(thetas[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(self.n_qubits - 1))

        self.circuit = circuit

    def run(self, thetas: Iterable[float] | np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit and return a NumPy array.

        Parameters
        ----------
        thetas : Iterable[float] or np.ndarray
            Parameter vector of length ``n_qubits``.

        Returns
        -------
        np.ndarray
            Expectation value of the last qubit wrapped in an array.
        """
        thetas = np.asarray(thetas, dtype=np.float32)
        if thetas.shape!= (self.n_qubits,):
            raise ValueError(f"Expected {self.n_qubits} parameters, got {thetas.shape}")
        expval = self.circuit(thetas)
        return np.array([expval])

    def gradient(self, thetas: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the output with respect to the inputs.

        Parameters
        ----------
        thetas : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Gradient vector of shape ``(n_qubits,)``.
        """
        grad_fn = qml.grad(self.circuit)
        return grad_fn(thetas)

    def train_step(
        self,
        thetas: np.ndarray,
        target: float,
        lr: float = 0.01,
        loss_fn: callable = lambda pred, tgt: (pred - tgt) ** 2,
    ) -> Tuple[float, np.ndarray]:
        """
        Perform a single training step using gradient descent.

        Parameters
        ----------
        thetas : np.ndarray
            Current parameters.
        target : float
            Desired expectation value.
        lr : float, default=0.01
            Learning rate.
        loss_fn : callable
            Loss function that takes (prediction, target).

        Returns
        -------
        loss_val : float
            Loss value.
        updated_thetas : np.ndarray
            Updated parameter vector.
        """
        pred = self.circuit(thetas)
        loss = loss_fn(pred, target)
        grads = qml.grad(self.circuit)(thetas)
        updated = thetas - lr * grads
        return float(loss), updated


def FCL(n_qubits: int = 1) -> QuantumFullyConnectedLayer:
    """
    Factory that returns a variational quantum layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.

    Returns
    -------
    QuantumFullyConnectedLayer
        Initialized quantum layer ready for inference or training.
    """
    return QuantumFullyConnectedLayer(n_qubits)


__all__ = ["FCL"]
