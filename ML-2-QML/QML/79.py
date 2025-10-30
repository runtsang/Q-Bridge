# FCL__gen087.py – Quantum implementation
"""Variational quantum fully‑connected layer with training capability.

This module extends the original single‑qubit example to a multi‑qubit
Pennylane circuit.  It supports a trainable parameter vector, an
expectation‑value readout, and a simple gradient‑descent optimiser.
The public API mirrors the seed – ``FCL()`` returns an object with
``run`` and ``train`` methods – so legacy code continues to function.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Iterable, Sequence


class QuantumFullyConnectedLayer:
    """Variational circuit that mimics a classical fully‑connected layer."""

    def __init__(self, n_qubits: int = 4, dev: qml.Device | None = None) -> None:
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits, shots=1024)
        self.wires = list(range(n_qubits))
        self._params: np.ndarray | None = None
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev)
        def circuit(params: Sequence[float]):
            # Encode parameters as rotations
            for i, w in enumerate(self.wires):
                qml.RY(params[i], w)
            # Entangle all qubits in a simple chain
            for i in range(self.n_qubits - 1):
                qml.CNOT(self.wires[i], self.wires[i + 1])
            # Readout expectation of Z on the first qubit
            return qml.expval(qml.PauliZ(self.wires[0]))
        return circuit

    def set_params(self, params: Sequence[float]) -> None:
        self._params = np.array(params, dtype=np.float32)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return expectation value of Z on the first qubit for the supplied parameters."""
        self.set_params(thetas)
        expval = self._circuit(self._params)
        return np.array([expval])

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 200,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Gradient‑based training of the variational parameters to fit data."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        # Initialise parameters randomly
        params = np.random.uniform(-np.pi, np.pi, self.n_qubits)

        for _ in range(epochs):
            for i in range(0, len(X), batch_size):
                xb = X[i : i + batch_size]
                yb = y[i : i + batch_size]
                # Loss is mean squared error between circuit output and targets
                loss_fn = lambda p, xb=xb, yb=yb: np.mean(
                    [(self._circuit(p) - yi) ** 2 for yi in yb]
                )
                params, _ = opt.step_and_cost(loss_fn, params)

        self.set_params(params)
        return self.run(params)


def FCL() -> QuantumFullyConnectedLayer:
    """Return a variational quantum fully‑connected layer."""
    return QuantumFullyConnectedLayer()


__all__ = ["FCL"]
