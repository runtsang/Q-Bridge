"""Hybrid quantum self‑attention with convolutional preprocessing.

The module builds a variational circuit that first encodes the input
through a small quanvolution block (a 2×2 kernel) and then applies
a self‑attention style entanglement across the token qubits.  The
output is a probability distribution over measurement outcomes
which can be interpreted as attention weights.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector

class SelfAttentionConv:
    """Quantum hybrid self‑attention block with convolutional preprocessing."""
    def __init__(self, n_qubits: int = 4, kernel_size: int = 2, threshold: float = 0.5):
        self.n_qubits = n_qubits
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)
        # Parameters for rotations (embed_dim * 3)
        self.rotation_params = ParameterVector("rot", length=n_qubits * 3)
        # Parameters for entangling gates (n_qubits-1)
        self.entangle_params = ParameterVector("ent", length=n_qubits - 1)
        # Parameters for quanvolution (kernel_size^2)
        self.quanv_params = ParameterVector("quanv", length=kernel_size ** 2)

    def _build_circuit(self, rotation_vals, entangle_vals, quanv_vals, data):
        """Construct a circuit for a single data instance."""
        circ = QuantumCircuit(self.qr, self.cr)
        # Convolutional encoding on first kernel_size^2 qubits
        for i in range(self.kernel_size ** 2):
            angle = quanv_vals[i]
            # Encode data threshold
            if data[i] > self.threshold:
                circ.rx(np.pi, i)
            else:
                circ.rx(0, i)
            circ.rx(angle, i)
        circ.barrier()
        # Attention rotation gates
        for i in range(self.n_qubits):
            circ.rx(rotation_vals[3 * i], i)
            circ.ry(rotation_vals[3 * i + 1], i)
            circ.rz(rotation_vals[3 * i + 2], i)
        circ.barrier()
        # Entangling gates
        for i in range(self.n_qubits - 1):
            circ.cx(i, i + 1)
            circ.rz(entangle_vals[i], i + 1)
        circ.barrier()
        circ.measure_all()
        return circ

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        data: np.ndarray,
        shots: int = 1024,
    ):
        """
        Execute the hybrid attention circuit on a backend.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Quantum backend to execute the circuit on.
        rotation_params : np.ndarray
            Shape (n_qubits, 3) – rotation angles for each qubit.
        entangle_params : np.ndarray
            Shape (n_qubits-1,) – angles for entangling gates.
        data : np.ndarray
            1‑D array of length kernel_size^2 containing classical values.
        shots : int
            Number of measurement shots.

        Returns
        -------
        dict
            Measurement counts.
        """
        # Flatten parameters
        rot = rotation_params.reshape(-1)
        ent = entangle_params.reshape(-1)
        quanv = np.linspace(0, np.pi / 2, self.kernel_size ** 2)
        circ = self._build_circuit(rot, ent, quanv, data)
        job = qiskit.execute(circ, backend, shots=shots)
        return job.result().get_counts(circ)

def SelfAttentionConv() -> SelfAttentionConv:
    """Factory that returns a hybrid quantum self‑attention instance."""
    backend = qiskit.Aer.get_backend("qasm_simulator")
    return SelfAttentionConv(n_qubits=4, kernel_size=2, threshold=0.5)

__all__ = ["SelfAttentionConv"]
