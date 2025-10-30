import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler

class HybridKernelAutoencoder:
    """
    Quantum kernel that compresses input data into a latent subspace via a
    RealAmplitudes ansatz and evaluates the overlap using a swap‑test.
    """
    def __init__(self, latent_dim: int = 3, num_trash: int = 2, reps: int = 5, backend=None):
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.n_qubits = latent_dim + num_trash
        self.ansatz = RealAmplitudes(self.n_qubits, reps=reps)
        self.sampler = Sampler(backend=backend)

    def _swap_test_circuit(self, x: np.ndarray, y: np.ndarray) -> QuantumCircuit:
        """Build a swap‑test circuit that compares two encoded states."""
        total_qubits = 2 * self.n_qubits + 1
        ancilla = total_qubits - 1
        qc = QuantumCircuit(total_qubits)

        # Encode first state into qubits 0..n-1
        qc.compose(self.ansatz, range(self.n_qubits), inplace=True, params=x)
        # Encode second state into qubits n..2n-1
        qc.compose(self.ansatz, range(self.n_qubits, 2 * self.n_qubits), inplace=True, params=y)

        # Swap‑test
        qc.h(ancilla)
        for i in range(self.n_qubits):
            qc.cswap(ancilla, i, self.n_qubits + i)
        qc.h(ancilla)

        qc.measure(ancilla, 0)
        return qc

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the swap‑test overlap between two classical vectors."""
        qc = self._swap_test_circuit(x, y)
        result = self.sampler.run(qc, shots=1024).result()
        counts = result.get_counts()
        p0 = counts.get("0", 0) / 1024
        return 2 * p0 - 1

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute Gram matrix between two sets of samples."""
        mat = np.empty((len(a), len(b)), dtype=np.float64)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = self.kernel(x, y)
        return mat

__all__ = ["HybridKernelAutoencoder"]
