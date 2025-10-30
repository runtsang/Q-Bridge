import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class SelfAttention:
    """
    Quantum self‑attention module using a variational circuit.
    """

    def __init__(self, n_qubits: int):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits used to represent the attention space.
        """
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Build a variational circuit that encodes rotation and entangle parameters.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for RX, RY, RZ gates, shape (3 * n_qubits,).
        entangle_params : np.ndarray
            Entanglement rotation angles, shape (n_qubits - 1,).

        Returns
        -------
        QuantumCircuit
            The constructed circuit.
        """
        circuit = QuantumCircuit(self.qr, self.cr)
        # Layer of single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entangling layer with parameterized RZ
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.rz(entangle_params[i], i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return a probability distribution over attention weights.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for RX, RY, RZ gates, shape (3 * n_qubits,).
        entangle_params : np.ndarray
            Entanglement rotation angles, shape (n_qubits - 1,).
        shots : int, optional
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Probability distribution of shape (2 ** n_qubits,).
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        # Convert counts to probabilities
        probs = np.array([counts.get(format(i, f'0{self.n_qubits}b'), 0) for i in range(2 ** self.n_qubits)]) / shots
        return probs
