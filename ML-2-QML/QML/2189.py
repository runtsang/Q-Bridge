"""Advanced self‑attention quantum circuit using PennyLane.

Features
--------
* Parameterized rotations for each qubit (rotation_params)
* Entangling gates controlled by entangle_params
* Supports state‑vector simulation or sampling from a qasm backend
"""

import pennylane as qml
import numpy as np
from typing import Optional, Tuple


class AdvancedSelfAttention:
    """
    Quantum self‑attention block implemented as a variational circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits representing the embedding dimension.
    device : str or qml.Device, optional
        PennyLane device; defaults to a state‑vector simulator.
    """

    def __init__(self, n_qubits: int, device: Optional[qml.Device] = None) -> None:
        self.n_qubits = n_qubits
        self.dev = device or qml.device("default.qubit", wires=n_qubits)

    def _circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> qml.Operation:
        """
        Builds the variational circuit used for attention.
        """
        @qml.qnode(self.dev)
        def circuit():
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entanglement layer (controlled RX)
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Measurement
            return qml.expval(qml.PauliZ(wires=range(self.n_qubits)))
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and return the expectation values.

        Parameters
        ----------
        rotation_params : array_like
            Parameters for R{X,Y,Z} rotations.
        entangle_params : array_like
            Parameters for controlled RX gates.
        shots : int, optional
            Number of measurement shots when using a qasm simulator.

        Returns
        -------
        ndarray
            Either the expectation values (state‑vector device) or a histogram
            of measurement outcomes (qasm device).
        """
        # Default to qasm simulator if shots requested > 0
        if shots > 0:
            self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)
        circuit = self._circuit(rotation_params, entangle_params)
        result = circuit()
        return np.array(result) if isinstance(result, (float, complex)) else result

__all__ = ["AdvancedSelfAttention"]
