"""QML module providing a simple variational circuit for a single qubit.

The circuit implements a single parameter rotation Ry(theta) after an initial
Hadamard gate. The expectation value of Z is returned as the output.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class QuantumCircuit:
    """Parameterised quantum circuit for a single qubit."""

    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(theta):
            qml.Hadamard(wires=0)
            qml.RY(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Run the circuit for each theta in thetas.

        Parameters
        ----------
        thetas : np.ndarray
            1‑D array of rotation angles.

        Returns
        -------
        np.ndarray
            Expectation values for each theta.
        """
        outputs = []
        for theta in thetas:
            val = self._circuit(pnp.array(theta))
            outputs.append(val)
        return np.array(outputs)
