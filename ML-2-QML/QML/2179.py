import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import qiskit

class SelfAttentionQuantum:
    """
    Quantum self‑attention using Pennylane on a Qiskit backend.
    Implements a two‑layer variational circuit:
    1. Rotation layer with RX, RY, RZ per qubit.
    2. Entangling layer with CRX gates between neighbouring qubits.
    The output is a probability distribution over measurement bit‑strings.
    """
    def __init__(self, n_qubits: int = 4, shots: int = 8192):
        self.n_qubits = n_qubits
        self.shots = shots
        # Default simulator; can be overridden via run()
        self.dev = qml.device("qiskit.aer", wires=n_qubits, shots=shots)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.dev, interface="jax")
        def circuit():
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])
            return qml.probs(wires=range(self.n_qubits))
        return circuit

    def run(
        self,
        backend: qiskit.providers.BaseBackend | None = None,
        rotation_params: np.ndarray = None,
        entangle_params: np.ndarray = None,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the variational circuit and return measurement probabilities.
        Parameters
        ----------
        backend : qiskit.providers.BaseBackend or None
            Optional Qiskit backend; if None, the Aer simulator is used.
        rotation_params : np.ndarray
            Array of length 3 * n_qubits specifying RX, RY, RZ angles per qubit.
        entangle_params : np.ndarray
            Array of length n_qubits - 1 specifying CRX angles.
        shots : int
            Number of shots for the simulation.
        Returns
        -------
        np.ndarray
            Probability distribution over bit‑strings of shape (2**n_qubits,).
        """
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        # Re‑create device with requested backend/shots
        self.dev = qml.device("qiskit.aer", wires=self.n_qubits, shots=shots, backend=backend)
        circuit = self._build_circuit(rotation_params, entangle_params)
        return circuit()

__all__ = ["SelfAttentionQuantum"]
