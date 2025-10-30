import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate

class SelfAttentionEnhanced:
    """
    Quantum self‑attention circuit with parameterized rotations and controlled‑phase gates.
    Supports execution on simulators or real devices with optional noise model.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Apply per‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Apply entangling controlled‑phase gates
        for i in range(self.n_qubits - 1):
            circuit.append(RZZGate(entangle_params[i]), [i, i + 1])
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()
        return result.get_counts(circuit)

# Example usage:
# backend = Aer.get_backend('qasm_simulator')
# attention = SelfAttentionEnhanced(n_qubits=4)
# counts = attention.run(backend, rotation_params=np.random.rand(12), entangle_params=np.random.rand(3))
