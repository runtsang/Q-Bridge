import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class SelfAttention:
    """
    Quantum self‑attention block implemented with a parameterized variational circuit.
    Rotations (RX, RY, RZ) are applied per qubit followed by a chain of controlled‑X entanglement.
    The run method returns Pauli‑Z expectation values for each qubit.
    """

    def __init__(self, n_qubits: int = 4, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend('qasm_simulator')

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits)
        cr = ClassicalRegister(self.n_qubits)
        circuit = QuantumCircuit(qr, cr)
        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], qr[i])
            circuit.ry(rotation_params[3 * i + 1], qr[i])
            circuit.rz(rotation_params[3 * i + 2], qr[i])
        # Entangling layer
        for i in range(self.n_qubits - 1):
            circuit.cx(qr[i], qr[i + 1])
            circuit.rz(entangle_params[i], qr[i + 1])
        circuit.measure(qr, cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        # Compute Pauli‑Z expectation values from measurement counts
        exp_vals = np.zeros(self.n_qubits)
        for state, cnt in counts.items():
            prob = cnt / shots
            bits = list(map(int, state[::-1]))  # little‑endian
            for idx, bit in enumerate(bits):
                exp = 1.0 if bit == 0 else -1.0
                exp_vals[idx] += exp * prob
        return exp_vals
