"""Quantum hybrid self‑attention block that combines a quantum convolution
   (inspired by the quanvolution filter) with a parameterised attention
   circuit.  The circuit is built with Qiskit and executed on an Aer
   simulator.  The output is a classical tensor that can be concatenated
   with other model components."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli

class QuantumSelfAttentionHybrid:
    """
    Quantum self‑attention that uses a 4‑qubit circuit to encode each 2×2 image
    patch, applies a random entangling layer, and measures Pauli‑Z to obtain
    a 4‑dimensional feature vector.  These vectors are weighted by a softmax
    over a learned score vector to produce a single attention‑aggregated
    feature per image.
    """

    def __init__(self, n_qubits: int = 4, patch_size: int = 2, image_size: int = 28):
        self.n_qubits = n_qubits
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_patches = (image_size // patch_size) ** 2
        self.backend = Aer.get_backend("qasm_simulator")
        # Learned score vector (initialized randomly; in practice should be trainable)
        self.score_vector = np.random.rand(n_qubits)

    def _encode_patch(self, qc: QuantumCircuit, patch: np.ndarray):
        """
        Encode a 2×2 patch (4 values) into Ry rotations on the qubits.
        """
        for i, val in enumerate(patch):
            qc.ry(val, i)

    def _measure_expectations(self, qc: QuantumCircuit, shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return the expectation values of Pauli‑Z on each qubit.
        """
        job = execute(qc, self.backend, shots=shots)
        counts = job.result().get_counts(qc)
        expvals = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            for i in range(self.n_qubits):
                bit = int(bitstring[self.n_qubits - 1 - i])  # reverse order
                expvals[i] += ((-1) ** bit) * cnt
        expvals /= shots
        return expvals

    def run(self, images: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        images : np.ndarray
            Input of shape (batch, 1, H, W).  Each 2×2 patch is encoded into a
            4‑qubit circuit.

        Returns
        -------
        np.ndarray
            Aggregated attention features of shape (batch, n_qubits).
        """
        batch, _, H, W = images.shape
        results = []
        for b in range(batch):
            img = images[b, 0]
            patch_features = []
            # Extract 2×2 patches
            for r in range(0, H, self.patch_size):
                for c in range(0, W, self.patch_size):
                    patch = img[r:r + self.patch_size, c:c + self.patch_size].flatten()
                    qc = QuantumCircuit(self.n_qubits)
                    self._encode_patch(qc, patch)
                    # Random entangling layer
                    for i in range(self.n_qubits - 1):
                        qc.cx(i, i + 1)
                    exp = self._measure_expectations(qc)
                    patch_features.append(exp)
            patch_features = np.array(patch_features)  # (N, n_qubits)
            # Compute attention scores via softmax over dot product with score_vector
            scores = np.exp(patch_features @ self.score_vector)
            attn_weights = scores / scores.sum()
            weighted = (patch_features * attn_weights[:, None]).sum(axis=0)
            results.append(weighted)
        return np.stack(results, axis=0)

__all__ = ["QuantumSelfAttentionHybrid"]
