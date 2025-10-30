"""Variational fully‑connected layer implemented with Pennylane.

The circuit applies a layer of RY rotations followed by a simple
entanglement pattern.  The expectation value of the Pauli‑Z operator
on the first qubit is returned as the layer output.  A lightweight
training routine using the Adam optimizer is provided.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class FCL:
    """Variational fully‑connected quantum layer.

    Parameters
    ----------
    n_qubits : int, default 1
        Number of qubits in the circuit.
    device : pennylane.Device, optional
        Quantum device.  Defaults to 'default.qubit'.
    """

    def __init__(self, n_qubits: int = 1, device: qml.Device | None = None) -> None:
        self.n_qubits = n_qubits
        self.dev = device or qml.device("default.qubit", wires=n_qubits)
        self._params = pnp.zeros(n_qubits, requires_grad=True)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: Sequence[float]) -> float:
            # H on all qubits
            for w in range(self.n_qubits):
                qml.H(wires=w)
            # Parameterized rotations
            for w, theta in enumerate(params):
                qml.RY(theta, wires=w)
            # Simple entanglement chain
            for w in range(self.n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            # Expectation of Z on qubit 0
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the circuit for a given set of parameters.

        Parameters
        ----------
        thetas : iterable of float
            Parameters to bind to the RY gates.
        """
        params = pnp.array(list(thetas), requires_grad=False)
        return np.array([self._circuit(params)])

    def train(
        self,
        x: Sequence[Sequence[float]],
        y: Sequence[float],
        epochs: int = 200,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        """Gradient‑based training using Adam.

        Parameters
        ----------
        x : sequence of parameter vectors
        y : sequence of target values
        epochs : int
        lr : float
        verbose : bool
            If True, print loss every 20 epochs.
        """
        optimizer = qml.AdamOptimizer(stepsize=lr)

        for epoch in range(1, epochs + 1):
            def loss_fn(params):
                loss = 0.0
                for inp, target in zip(x, y):
                    pred = self._circuit(pnp.array(inp, requires_grad=True))
                    loss += (pred - target) ** 2
                return loss / len(x)

            self._params = optimizer.step(loss_fn, self._params)

            if verbose and epoch % 20 == 0:
                loss_val = loss_fn(self._params)
                print(f"Epoch {epoch:3d} | Loss: {loss_val:.6f}")

    def predict(self, x: Sequence[Sequence[float]]) -> np.ndarray:
        """Return predictions for a batch of inputs."""
        return np.array([self._circuit(pnp.array(inp, requires_grad=False)) for inp in x])

__all__ = ["FCL"]
