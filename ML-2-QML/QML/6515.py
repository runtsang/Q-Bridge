"""Quantum self‑attention built with Pennylane.

The circuit encodes the classical attention output via angle encoding, applies a
parameterized rotation layer, an entangling layer, and measures the expectation
values of Pauli‑Z on each qubit.  The interface keeps the original
``run(backend, rotation_params, entangle_params, inputs, shots)`` signature.
"""

import pennylane as qml
import pennylane.numpy as np
from typing import Any

class QuantumSelfAttention:
    """
    Variational quantum circuit that implements a self‑attention style block.
    """

    def __init__(self, n_qubits: int = 4, device_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=n_qubits)

    def _circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ):
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            # Angle encoding of the classical attention output
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.CRZ(entangle_params[i], wires=[i, i + 1])

            # Measurement: expectation of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit()

    def run(
        self,
        backend: Any,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the quantum self‑attention circuit.

        Parameters
        ----------
        backend : Any
            Unused in this simplified implementation; kept for API compatibility.
        rotation_params : np.ndarray
            Shape (3 * n_qubits,). Rotation angles for each qubit.
        entangle_params : np.ndarray
            Shape (n_qubits - 1,). Entanglement angles.
        inputs : np.ndarray
            Classical attention output to encode, shape (n_qubits,).
        shots : int
            Number of shots for the simulator (ignored for analytic device).

        Returns
        -------
        np.ndarray
            Expectation values of Pauli‑Z on each qubit.
        """
        # In a real deployment, 'backend' would control device selection.
        return self._circuit(rotation_params, entangle_params, inputs)

__all__ = ["QuantumSelfAttention"]
