"""
FCL__gen260.py – Quantum variational circuit with entanglement and adaptive measurement.
"""

import pennylane as qml
import pennylane.numpy as np
from typing import Iterable, List


def FCL() -> "FCL":
    """
    Return a variational quantum circuit that models a fully‑connected layer.
    The circuit supports a multi‑qubit ansatz with Ry rotations and CNOT entanglement.
    """

    class FCL:
        """
        Variational quantum circuit with parameterized Ry gates and a simple entanglement pattern.
        """

        def __init__(self, n_qubits: int = 2, shots: int = 1000) -> None:
            self.n_qubits = n_qubits
            self.device = qml.device("default.qubit", wires=n_qubits, shots=shots)

            @qml.qnode(self.device, interface="autograd")
            def circuit(params):
                # Parameterized Ry rotations on each qubit
                for i in range(n_qubits):
                    qml.RY(params[i], wires=i)
                # Entangling CNOT ladder
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Expectation of PauliZ on the first qubit
                return qml.expval(qml.PauliZ(0))

            self.circuit = circuit

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """
            Evaluate the circuit for each theta value.
            ``thetas`` is an iterable of scalars; each theta is broadcast to all qubits.
            Returns a NumPy array of expectation values.
            """
            params = np.array([thetas] * self.n_qubits)  # shape (n_qubits, len(thetas))
            # Transpose to (len(thetas), n_qubits) for broadcasting
            params = params.T
            expectations = []
            for param_vec in params:
                expectations.append(self.circuit(param_vec))
            return np.array(expectations)

    return FCL()
__all__ = ["FCL"]
