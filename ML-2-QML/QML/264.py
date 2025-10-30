import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class SelfAttention:
    """Quantum self‑attention built with a parameterised variational circuit.

    The circuit encodes one qubit per token.  Single‑qubit rotations
    and neighbouring CRX gates are parameterised by the supplied
    arrays.  Measurement outcomes are interpreted as a probability
    distribution that is reshaped into a square attention matrix.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        """Construct the variational circuit."""
        circuit = QuantumCircuit(self.qr, self.cr)

        for i in range(self.n_qubits):
            theta_x, theta_y, theta_z = rotation_params[i]
            circuit.rx(theta_x, i)
            circuit.ry(theta_y, i)
            circuit.rz(theta_z, i)

        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return a normalised attention matrix.

        Parameters
        ----------
        backend
            Qiskit backend (Aer simulator or real device).
        rotation_params : np.ndarray
            Array of shape (n_qubits, 3) containing RX,RY,RZ angles.
        entangle_params : np.ndarray
            Array of shape (n_qubits-1,) containing CRX angles.
        shots : int, optional
            Number of shots for the execution.

        Returns
        -------
        np.ndarray
            Attention matrix of shape (n_qubits, n_qubits) whose rows
            sum to 1.  Each row corresponds to a token and each
            column to a source token.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        probs = np.zeros(2 ** self.n_qubits, dtype=np.float64)
        for bitstring, count in counts.items():
            idx = int(bitstring[::-1], 2)  # little‑endian
            probs[idx] = count / shots

        # Interpret the first n_qubits bits as the source token
        # and the remaining as the target token.
        attention = probs.reshape((2 ** self.n_qubits,)).reshape(
            (self.n_qubits, 2 ** (self.n_qubits - 1))
        )

        # Normalise rows to obtain a soft‑max‑like attention map
        attention = attention / attention.sum(axis=1, keepdims=True)
        return attention

# Default instance for convenience
backend = Aer.get_backend("qasm_simulator")
attention = SelfAttention(n_qubits=4)
