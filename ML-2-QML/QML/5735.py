"""Quantum variational circuit for a fully‑connected layer with batched execution."""

import pennylane as qml
import numpy as np

class QuantumCircuit:
    """
    Parameterized quantum circuit that emulates a fully‑connected layer
    by encoding input features into qubit rotations and measuring
    expectation values of a Pauli‑Z observable.
    """

    def __init__(self, n_qubits: int, device_name: str = "default.qubit", shots: int = 1000):
        self.n_qubits = n_qubits
        self.device = qml.device(device_name, wires=n_qubits, shots=shots)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device, interface="autograd")
        def circuit(params, x):
            # Encode input features
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            # Parameterized rotations
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measurement
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, params: np.ndarray, batch: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of inputs.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector of shape (n_qubits,).
        batch : np.ndarray
            Input batch of shape (batch_size, n_qubits).

        Returns
        -------
        np.ndarray
            Array of expectation values with shape (batch_size,).
        """
        expectations = []
        for x in batch:
            expectations.append(self.circuit(params, x))
        return np.array(expectations)

__all__ = ["QuantumCircuit"]
