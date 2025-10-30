import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class QuantumSelfAttention:
    """Variational quantum circuit that outputs a probability distribution
    over `n_qubits` measurement outcomes. The distribution is interpreted
    as attention weights for a single head.
    """

    def __init__(self, n_qubits: int, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        # Apply parameterised rotations
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)

        # Entangle neighbouring qubits
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(entangle_params[i], i + 1)

        qc.measure(qr, cr)
        return qc

    def run(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> np.ndarray:
        """
        Execute the circuit and return a probability vector of shape
        (n_qubits,). The probabilities sum to 1.0 and can be used as
        attention weights for a single head.
        """
        qc = self._build_circuit(rotation_params, entangle_params)
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(qc)

        probs = np.zeros(self.n_qubits)
        total = sum(counts.values())
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            probs[idx] += count / total
        return probs

__all__ = ["QuantumSelfAttention"]
