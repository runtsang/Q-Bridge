import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import numpy as np

class SelfAttention:
    """
    Quantum self‑attention block using Qiskit. The circuit parameters are
    split into rotation and entanglement layers, matching the classical
    interface ``run(rotation_params, entangle_params, inputs)``.
    """

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray):
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Basis encoding of inputs
        for i in range(self.n_qubits):
            if inputs.flatten()[i] > 0.5:
                circuit.x(qr[i])

        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], qr[i])
            circuit.ry(rotation_params[3 * i + 1], qr[i])
            circuit.rz(rotation_params[3 * i + 2], qr[i])

        # Entanglement layer
        for i in range(self.n_qubits - 1):
            circuit.cnot(qr[i], qr[i + 1])
            circuit.rz(entangle_params[i], qr[i])
            circuit.cnot(qr[i], qr[i + 1])

        circuit.measure(qr, cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params, inputs)
        job = execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)
        # Convert counts to probability distribution over basis states
        probs = np.zeros(2 ** self.n_qubits)
        for state, count in counts.items():
            idx = int(state[::-1], 2)  # Qiskit returns little‑endian
            probs[idx] = count / shots
        return probs

__all__ = ["SelfAttention"]
