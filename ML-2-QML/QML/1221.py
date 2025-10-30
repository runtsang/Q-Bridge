"""Variational quantum fully‑connected layer with parameter‑shift gradients.

The quantum version replaces the single‑qubit rotation of the seed with a
multi‑qubit entangled circuit.  Parameters are supplied as a flat list,
matching the classical network’s weight vector.  The circuit returns the
expectation value of the Pauli‑Z operator on the last qubit, which is
interpreted as the layer output.  A lightweight training helper is also
included.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class FCLModelQuantum:
    """
    Variational circuit that emulates a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.  This corresponds to the number of
        parameters in the weight matrix of the classical network.
    """
    def __init__(self, n_qubits: int = 1) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=1024)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: List[float]) -> float:
            # Encode each parameter as a rotation about Y on a separate qubit.
            for i, theta in enumerate(params):
                qml.RY(theta, wires=i)

            # Simple entanglement pattern (chain).
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Measurement of the last qubit in the Z basis.
            return qml.expval(qml.PauliZ(self.n_qubits - 1))

        self.circuit = circuit

    # ------------------------------------------------------------------
    # Run interface (keeps compatibility with the seed)
    # ------------------------------------------------------------------
    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit on a list of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat list of rotation angles.  The list length must match
            `self.n_qubits`; otherwise a ValueError is raised.
        """
        params = np.array(list(thetas), dtype=float)
        if params.size!= self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} parameters, got {params.size}."
            )
        expectation = self.circuit(params)
        return np.array([expectation])

    # ------------------------------------------------------------------
    # Gradient helper (parameter‑shift rule)
    # ------------------------------------------------------------------
    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the expectation value w.r.t. the parameters
        using the parameter‑shift rule.

        Returns
        -------
        grads : np.ndarray
            Gradient vector of shape `(n_qubits,)`.
        """
        params = pnp.array(list(thetas), dtype=float)
        grads = qml.grad(self.circuit)(params)
        return grads

    # ------------------------------------------------------------------
    # Simple training loop (illustrative)
    # ------------------------------------------------------------------
    def train(
        self,
        data: List[List[float]],
        targets: List[float],
        lr: float = 0.01,
        epochs: int = 100,
    ) -> None:
        """
        Gradient‑descent training on a small dataset.

        Parameters
        ----------
        data : List[List[float]]
            List of parameter vectors (each of length `n_qubits`).
        targets : List[float]
            Corresponding target values.
        lr : float, optional
            Learning rate.
        epochs : int, optional
            Number of epochs.
        """
        params = pnp.random.uniform(-np.pi, np.pi, size=self.n_qubits)

        for epoch in range(epochs):
            loss = 0.0
            grads = np.zeros_like(params)
            for x, y in zip(data, targets):
                y_pred = self.circuit(params)
                loss += (y_pred - y) ** 2
                grads += 2 * (y_pred - y) * qml.grad(self.circuit)(params)
            loss /= len(data)
            grads /= len(data)

            params -= lr * grads

        # Store the trained parameters for later use.
        self.trained_params = params

        # Optionally expose a `run` method that uses the trained weights.
        def _run(thetas: Iterable[float]) -> np.ndarray:
            return self.circuit(self.trained_params).numpy()

        self.run = _run

__all__ = ["FCLModelQuantum"]
