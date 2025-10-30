import pennylane as qml
import numpy as np
import qiskit
from qiskit import execute
from pennylane.backends.qiskit import QiskitBackend

class QuantumSelfAttentionGen013:
    """Variational quantum self‑attention block built with Pennylane.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used in the circuit.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        # Default Pennylane device; can be overridden by the run method
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Construct the variational circuit used for self‑attention."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entanglement layer (nearest‑neighbour CRX)
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        """
        Execute the variational circuit on the supplied backend.

        Parameters
        ----------
        backend : str or qiskit.providers.ibmq.IBMQBackend
            Target backend name (e.g. "default.qubit") or an instantiated Qiskit backend.
        rotation_params : np.ndarray
            Rotation angles, shape (3 * n_qubits,).
        entangle_params : np.ndarray
            Entanglement angles, shape (n_qubits - 1,).
        shots : int, optional
            Number of measurement shots when running on a Qiskit backend. Default: 1024.

        Returns
        -------
        dict or list
            For a Pennylane device, a list of expectation values per qubit.
            For a Qiskit backend, a measurement count dictionary.
        """
        # Decide on device
        if isinstance(backend, str):
            # Use Pennylane device
            self.dev = qml.device(backend, wires=self.n_qubits)
            circuit = self._circuit(rotation_params, entangle_params)
            return circuit()

        # Assume a Qiskit backend: convert the Pennylane circuit to a Qiskit circuit
        circuit = self._circuit(rotation_params, entangle_params)
        qiskit_circuit = QiskitBackend().from_qnode(circuit)
        result = execute(qiskit_circuit, backend, shots=shots).result()
        return result.get_counts(qiskit_circuit)

__all__ = ["QuantumSelfAttentionGen013"]
