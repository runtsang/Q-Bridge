"""Variational quantum fully‑connected layer using Pennylane."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class FCLGen035:
    """
    Variational quantum circuit that emulates a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, which also determines the input dimensionality.
    layers : int
        Number of variational layers.
    device : str | qml.Device, optional
        Pennylane device to execute the circuit. Defaults to a GPU‑enabled
        default.qubit if available, otherwise a CPU simulator.
    shots : int, optional
        Number of shots for expectation estimation. If ``None`` the device
        computes the expectation analytically.
    """
    def __init__(
        self,
        n_qubits: int = 1,
        layers: int = 1,
        device: str | qml.Device | None = None,
        shots: int | None = None,
    ) -> None:
        if device is None:
            try:
                device = qml.device("default.qubit", wires=n_qubits, shots=shots)
            except Exception:
                device = qml.device("default.qubit", wires=n_qubits)
        self._dev = device
        self.n_qubits = n_qubits
        self.layers = layers
        self._theta = pnp.array(
            np.random.uniform(0, 2 * np.pi, size=(layers, n_qubits)),
            requires_grad=False,
        )

    def _circuit(self, thetas: pnp.ndarray):
        """Parameterized circuit used by the QNode."""
        for i in range(self.layers):
            for j in range(self.n_qubits):
                qml.RY(thetas[i, j], wires=j)
            if i < self.layers - 1:
                for j in range(self.n_qubits - 1):
                    qml.CNOT(wires=[j, j + 1])

        return qml.expval(qml.PauliZ(0))

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the variational circuit for a batch of input angles.

        Parameters
        ----------
        thetas : np.ndarray, shape (batch, n_qubits)
            Input angles for each qubit.

        Returns
        -------
        np.ndarray
            Array of expectation values, shape (batch,).
        """
        # Convert to Pennylane numpy array
        thetas = pnp.array(thetas, dtype=pnp.float64)
        qnode = qml.QNode(self._circuit, self._dev, interface="autograd")
        expectations = []
        for theta in thetas:
            expectations.append(qnode(theta))
        return np.array(expectations)

def FCL() -> FCLGen035:
    """Return an instance of the quantum fully‑connected layer."""
    return FCLGen035(n_qubits=1, layers=2)

__all__ = ["FCL", "FCLGen035"]
