"""Quantum fully‑connected layer using Pennylane.

The circuit implements a depth‑controlled variational ansatz with
entanglement across all qubits.  It exposes a ``run`` method that
returns the expectation value of Pauli‑Z on the first qubit and a
``grad`` method that returns the analytic gradient via Pennylane's
parameter‑shift rule.  A minimal ``train_step`` helper demonstrates
how to update the circuit parameters using a classic optimizer.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer


class QuantumFullyConnectedLayer:
    """Variational circuit emulating a fully‑connected layer.

    Parameters
    ----------
    n_qubits: int
        Number of qubits in the circuit.  Each qubit represents a
        feature dimension.
    depth: int
        Number of repeated rotation layers.
    shots: int, optional
        Number of shots for the simulator.  Default: 1024.
    dev_name: str, optional
        Pennylane device name.  Default: ``"default.qubit"``.
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int,
        shots: int = 1024,
        dev_name: str = "default.qubit",
    ) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.device = qml.device(dev_name, wires=n_qubits, shots=shots)
        # Parameters are stored as a (depth, n_qubits) array
        self.params = pnp.random.randn(depth, n_qubits)
        self.qnode = qml.QNode(self._circuit, self.device)

    # ------------------------------------------------------------------
    # Circuit definition --------------------------------------------------
    # ------------------------------------------------------------------
    def _circuit(self, params: np.ndarray) -> qml.expval:
        for i in range(self.depth):
            for j in range(self.n_qubits):
                qml.RY(params[i, j], wires=j)
            # Entangle all qubits with a ring of CNOTs
            for j in range(self.n_qubits):
                qml.CNOT(wires=[j, (j + 1) % self.n_qubits])
        return qml.expval(qml.PauliZ(0))

    # ------------------------------------------------------------------
    # Public API ---------------------------------------------------------
    # ------------------------------------------------------------------
    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the expectation value for a single parameter vector.

        Parameters
        ----------
        thetas: Iterable[float]
            Flattened parameter vector of shape ``(depth, n_qubits)``.
        """
        params = np.array(thetas, dtype=float).reshape(self.depth, self.n_qubits)
        return np.array([self.qnode(params)])

    def grad(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the analytic gradient w.r.t. all parameters.

        Parameters
        ----------
        thetas: Iterable[float]
            Flattened parameter vector of shape ``(depth, n_qubits)``.
        """
        params = np.array(thetas, dtype=float).reshape(self.depth, self.n_qubits)
        return np.array(self.qnode.grad(params).flatten())

    def train_step(
        self,
        thetas: Iterable[float],
        target: float,
        learning_rate: float = 0.01,
        loss_fn: str = "mse",
    ) -> tuple[float, np.ndarray]:
        """Perform one gradient‑descent step and return loss and updated params.

        Parameters
        ----------
        thetas: Iterable[float]
            Current flattened parameter vector.
        target: float
            Desired expectation value.
        learning_rate: float, optional
            Step size for the Adam optimizer.
        loss_fn: str, optional
            Either ``"mse"`` or ``"mae"``.
        """
        params = np.array(thetas, dtype=float).reshape(self.depth, self.n_qubits)

        if loss_fn == "mse":
            loss = lambda p: (self.qnode(p) - target) ** 2
        elif loss_fn == "mae":
            loss = lambda p: np.abs(self.qnode(p) - target)
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")

        optimizer = AdamOptimizer(stepsize=learning_rate)
        new_params, _ = optimizer.step(loss, params)
        loss_val = float(loss(new_params))
        return loss_val, new_params.flatten()

    def get_params(self) -> np.ndarray:
        """Return the flattened parameter array."""
        return self.params.flatten()

    def set_params(self, params: Iterable[float]) -> None:
        """Set the circuit parameters from a flattened array."""
        self.params = np.array(params, dtype=float).reshape(self.depth, self.n_qubits)


def FCL(n_qubits: int = 1, depth: int = 3) -> QuantumFullyConnectedLayer:
    """Return a quantum fully‑connected layer instance.

    The default configuration reproduces the original seed but can be
    tuned via ``n_qubits`` and ``depth``.
    """
    return QuantumFullyConnectedLayer(n_qubits, depth)


__all__ = ["FCL"]
