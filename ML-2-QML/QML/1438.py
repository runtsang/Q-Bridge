"""Quantum self‑attention using PennyLane variational circuits.

The class shares the same API as its classical counterpart: ``run(rotation_params, entangle_params, inputs)``.
"""

import pennylane as qml
import numpy as np

class SelfAttentionBlock:
    """
    Variational quantum circuit that emulates a self‑attention block.
    ``inputs`` are ignored; only the parameters shape the circuit state.
    """

    def __init__(self, n_qubits: int):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit.
        """
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.dev)
        def circuit():
            # Single‑qubit rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                if i < len(entangle_params):
                    qml.RZ(entangle_params[i], wires=i + 1)
            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray = None,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the variational circuit and return expectation values.

        Parameters
        ----------
        rotation_params : array-like
            Parameters for RX, RY, RZ gates. Shape: (3 * n_qubits,).
        entangle_params : array-like
            Optional parameters for additional entanglement.
        inputs : array-like, optional
            Ignored in the quantum implementation but kept for API compatibility.
        shots : int, optional
            Number of shots for sampling; used only if a simulator that supports sampling is chosen.

        Returns
        -------
        output : ndarray
            Vector of expectation values of shape (n_qubits,).
        """
        circuit = self._circuit(rotation_params, entangle_params)
        return np.array(circuit())
