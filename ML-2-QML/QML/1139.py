"""
Quantum fully connected layer using Pennylane.

The circuit implements a parameterized variational ansatz with alternating
single‑qubit rotations and entangling CNOT layers.  The number of qubits and
layers are configurable, allowing the model to scale with problem size.
The `run` method evaluates the expectation value of a Pauli‑Z observable
on the first qubit, returning it as a NumPy array.

Usage
-----
>>> from FCL__gen072 import FCL
>>> qc = FCL(n_qubits=3, n_layers=2, shots=1024)
>>> params = qc.get_flat_params()
>>> out = qc.run(params)
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Iterable, List


class QuantumCircuit:
    """
    Variational quantum circuit emulating a fully connected layer.
    """

    def __init__(self, n_qubits: int = 1, n_layers: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        # Create a flat list of parameters for all rotation angles
        self.param_shape = (n_layers, n_qubits, 3)  # RX, RY, RZ per qubit per layer
        self.n_params = np.prod(self.param_shape)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # Unpack parameters
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.RX(params[l, q, 0], wires=q)
                    qml.RY(params[l, q, 1], wires=q)
                    qml.RZ(params[l, q, 2], wires=q)
                # Entangling layer
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                # Wrap around entanglement
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            # Measurement: expectation of Pauli‑Z on first qubit
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit given a flat list of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flattened rotation angles for all layers and qubits.

        Returns
        -------
        np.ndarray
            Expectation value of the first qubit as a 1‑D array.
        """
        flat = np.array(list(thetas), dtype=np.float64)
        if flat.size!= self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {flat.size}"
            )
        params = flat.reshape(self.param_shape)
        expectation = self._circuit(params)
        return np.array([expectation])

    def get_flat_params(self) -> np.ndarray:
        """
        Return a random initialization of parameters as a flat NumPy array.
        """
        return pnp.random.uniform(-np.pi, np.pi, size=self.n_params)

    def __call__(self, thetas: Iterable[float]) -> np.ndarray:
        return self.run(thetas)


__all__ = ["QuantumCircuit"]
