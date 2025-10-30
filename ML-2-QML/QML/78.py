import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class QuantumSelfAttention:
    """
    Variational quantum self‑attention block.
    The circuit applies rotation and entanglement gates according to
    the supplied parameters and measures the expectation of Pauli‑Z
    on each qubit.  The returned vector can be interpreted as
    attention weights for a downstream classical layer.
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entanglement layer
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
        Executes the attention circuit on the supplied backend and returns
        the expectation values of Pauli‑Z on each qubit.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        exp_vals = np.zeros(self.n_qubits)
        for i in range(self.n_qubits):
            exp = 0.0
            for bitstring, count in counts.items():
                bit = int(bitstring[self.n_qubits - 1 - i])
                exp += ((-1) ** bit) * count
            exp_vals[i] = exp / shots
        return exp_vals

def SelfAttention() -> QuantumSelfAttention:
    """
    Factory that returns a quantum self‑attention instance.
    """
    backend = Aer.get_backend("qasm_simulator")
    return QuantumSelfAttention(n_qubits=4)

__all__ = ["SelfAttention"]
