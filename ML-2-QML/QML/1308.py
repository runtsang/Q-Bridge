import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class QuantumAttention:
    """
    Generates a quantum attention matrix from rotation and entanglement parameters.
    The circuit consists of single‑qubit rotations followed by a chain of CNOT gates.
    The resulting measurement counts are used to compute a probability distribution
    over the qubits, which is then turned into an attention matrix.
    """
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """
        Build a circuit with rotations and a chain of CNOTs.
        rotation_params: shape (n_qubits, 3) for RX,RY,RZ angles.
        entangle_params: shape (n_qubits-1,) for CRX angles.
        """
        qr = QuantumRegister(self.n_qubits)
        cr = ClassicalRegister(self.n_qubits)
        circuit = QuantumCircuit(qr, cr)

        for i in range(self.n_qubits):
            circuit.rx(rotation_params[i, 0], qr[i])
            circuit.ry(rotation_params[i, 1], qr[i])
            circuit.rz(rotation_params[i, 2], qr[i])

        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], qr[i], qr[i+1])

        circuit.measure(qr, cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """
        Execute the circuit and return a probability vector over qubit states.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circuit)
        probs = np.zeros(self.n_qubits)
        for bitstring, count in counts.items():
            for i, bit in enumerate(reversed(bitstring)):
                probs[i] += count * int(bit)
        probs /= self.shots
        return probs

    def get_attention_matrix(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """
        Return a symmetric attention matrix derived from the probability vector.
        """
        probs = self.run(rotation_params, entangle_params)
        # Outer product to form a matrix, then normalize rows to sum to 1
        mat = np.outer(probs, probs)
        mat = mat / mat.sum(axis=1, keepdims=True)
        return mat

__all__ = ["QuantumAttention"]
