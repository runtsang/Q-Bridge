"""
FCLayer – Variational quantum circuit with automatic parameter‑shift gradients.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class FCLayer:
    """
    A parameterised quantum circuit that mimics a fully‑connected layer.
    Supports multiple qubits, a configurable ansatz, and automatic gradients.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    layers : int, optional
        Number of ansatz layers. Each layer consists of a layer of single‑qubit RY gates
        followed by a layer of CNOTs in a linear chain.
    device : str, optional
        Pennylane device name. Defaults to ``default.qubit``.
    shots : int, optional
        Number of shots for expectation estimation. If ``None`` the device is used in
        analytic mode.
    """

    def __init__(self, n_qubits: int, layers: int = 1, device: str = "default.qubit", shots: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.device = qml.device(device, wires=n_qubits, shots=shots)
        self.n_params = n_qubits * layers  # one RY per qubit per layer

        @qml.qnode(self.device, interface="autograd")
        def circuit(params: Sequence[float]) -> float:
            # State preparation – start in |0⟩
            for w in range(n_qubits):
                qml.Hadamard(w)

            # Ansatz
            for l in range(layers):
                for w in range(n_qubits):
                    qml.RY(params[l * n_qubits + w], w)
                # Entangling layer (linear chain)
                for w in range(n_qubits - 1):
                    qml.CNOT(w, w + 1)

            # Expectation value of Z on the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the variational circuit on a batch of parameter lists.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of parameter sequences. Each sequence must have length
            ``n_params``. They are stacked into a batch and processed
            sequentially.

        Returns
        -------
        np.ndarray
            Array of expectation values, one per parameter set.
        """
        thetas_arr = np.atleast_2d(np.array(list(thetas), dtype=np.float64))
        if thetas_arr.shape[1]!= self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters per sample, got {thetas_arr.shape[1]}")
        return np.array([self.circuit(params) for params in thetas_arr])

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the circuit output with respect to its parameters
        using the parameter‑shift rule.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of parameter sequences.

        Returns
        -------
        np.ndarray
            Gradient matrix of shape (batch, n_params).
        """
        thetas_arr = np.atleast_2d(np.array(list(thetas), dtype=np.float64))
        grads = []
        for params in thetas_arr:
            grads.append(qml.grad(self.circuit)(params))
        return np.array(grads)

    def train_step(
        self,
        thetas: Iterable[float],
        targets: Iterable[float],
        lr: float = 1e-3,
    ) -> float:
        """
        Perform one gradient‑descent update on the circuit parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Input parameter sets.
        targets : Iterable[float]
            Desired expectation values.
        lr : float
            Learning rate.

        Returns
        -------
        float
            Mean squared error loss after the update.
        """
        thetas_arr = np.atleast_2d(np.array(list(thetas), dtype=np.float64))
        targets_arr = np.array(list(targets), dtype=np.float64)

        # Forward pass
        preds = self.run(thetas_arr)

        # Compute gradients
        grads = self.gradient(thetas_arr)

        # Update parameters (simple SGD)
        for i, params in enumerate(thetas_arr):
            thetas_arr[i] -= lr * grads[i]

        # Return loss
        loss = np.mean((preds - targets_arr) ** 2)
        return loss


__all__ = ["FCLayer"]
