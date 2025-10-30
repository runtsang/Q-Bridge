import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class SelfAttentionImpl:
    """
    Quantum self‑attention circuit that encodes input features with a
    random layer and parameterized rotations, then measures all qubits.
    The interface matches the classical counterpart: run(backend,
    rotation_params, entangle_params, shots).
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _random_layer(self, circuit: QuantumCircuit) -> None:
        """
        Apply a random single‑qubit rotation to each qubit before the
        parameterized gates.  This mimics the RandomLayer from Quantum‑NAT.
        """
        for q in range(self.n_qubits):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            lam = np.random.uniform(0, 2*np.pi)
            circuit.u3(theta, phi, lam, q)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        self._random_layer(circuit)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

def SelfAttention():
    """
    Factory that returns a SelfAttentionImpl instance using the Aer qasm simulator.
    """
    backend = Aer.get_backend("qasm_simulator")
    return SelfAttentionImpl(n_qubits=4)

__all__ = ["SelfAttentionImpl", "SelfAttention"]
