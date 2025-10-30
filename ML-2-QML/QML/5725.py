import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class QuantumSelfAttention:
    """
    A Qiskit implementation of a self‑attention style block.
    Produces expectation values of Pauli‑Z on each qubit.
    """

    def __init__(self, n_qubits: int, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend('qasm_simulator')

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        circuit = QuantumCircuit(qr, cr)

        # Single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement between neighbours
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(qr, cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)

        # Convert counts to expectation values of Pauli‑Z
        exp = np.zeros(self.n_qubits, dtype=np.float32)
        for bitstring, cnt in counts.items():
            prob = cnt / shots
            for i, bit in enumerate(reversed(bitstring)):
                z = 1.0 if bit == '0' else -1.0
                exp[i] += z * prob
        return exp

__all__ = ["QuantumSelfAttention"]
