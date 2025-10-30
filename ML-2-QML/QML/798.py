"""FullyConnectedLayer: Parameterized quantum circuit for a fully connected layer.

The circuit uses n_qubits equal to the number of input features and applies a
layer of RY gates followed by a chain of CX entangling gates, then a second
layer of RZ gates. The expectation value of PauliZ on each qubit is returned
as a vector of length n_qubits. The circuit is wrapped in a PennyLane
QuantumNode for easy execution on a simulator or real device.
"""

import numpy as np
import pennylane as qml
from typing import Iterable


class FullyConnectedLayer:
    """
    Parameterized quantum circuit representing a fully connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (features).
    shots : int, optional
        Number of shots for sampling. 0 uses analytic expectation.
    """

    def __init__(self, n_qubits: int, shots: int = 0) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots or None)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # params shape: (2 * n_qubits,)
            # First layer of RY
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            # Entangling chain
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Second layer of RZ
            for i in range(n_qubits):
                qml.RZ(params[n_qubits + i], wires=i)
            # Measurement of PauliZ expectation on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flat list of parameters of length 2 * n_qubits (RY and RZ angles).

        Returns
        -------
        np.ndarray
            Expectation values of PauliZ on each qubit.
        """
        params = np.array(thetas, dtype=np.float32)
        if params.size!= 2 * self.n_qubits:
            raise ValueError(
                f"Expected {2 * self.n_qubits} parameters, got {params.size}"
            )
        result = self.circuit(params)
        return np.array(result)


__all__ = ["FullyConnectedLayer"]
