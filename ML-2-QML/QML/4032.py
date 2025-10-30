import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class HybridSelfAttentionQuanvolution:
    """Quantum module that fuses a quanvolution encoder with a self‑attention style circuit."""
    def __init__(self, n_qubits: int = 16):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, 'q')
        self.cr = ClassicalRegister(n_qubits, 'c')
        self.backend = qiskit.Aer.get_backend('qasm_simulator')

    def _encode_patches(self, circuit: QuantumCircuit, rotation_params: np.ndarray):
        """Encode 2x2 patches into qubits with given rotations."""
        # rotation_params shape: (n_qubits, 3)
        for i in range(self.n_qubits):
            rx, ry, rz = rotation_params[i]
            circuit.rx(rx, i)
            circuit.ry(ry, i)
            circuit.rz(rz, i)

    def _entangle(self, circuit: QuantumCircuit, entangle_params: np.ndarray):
        """Entangle neighbouring qubits according to entangle_params."""
        # entangle_params shape: (n_qubits-1,)
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.rz(entangle_params[i], i + 1)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024):
        """
        Build and execute the hybrid circuit.
        :param rotation_params: rotation angles for each qubit, shape (n_qubits, 3)
        :param entangle_params: entanglement angles, shape (n_qubits-1,)
        :return: measurement counts
        """
        circuit = QuantumCircuit(self.qr, self.cr)
        self._encode_patches(circuit, rotation_params)
        self._entangle(circuit, entangle_params)
        circuit.measure(self.qr, self.cr)
        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

__all__ = ["HybridSelfAttentionQuanvolution"]
