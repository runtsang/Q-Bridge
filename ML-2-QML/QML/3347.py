"""Quantum hybrid self‑attention module inspired by quanvolution.

The public API matches the classical counterpart: a function
`SelfAttention()` returns an instance of :class:`HybridSelfAttention`.
The class expects a number of qubits per patch (default 4) and a
patch size.  Inputs are image‑like arrays of shape
``(batch, channels, height, width)``.  For each patch a parameterised
circuit is built, encoded with the supplied rotation parameters,
entangled with the entangle parameters, and measured.  The resulting
measurement vectors are post‑processed classically to compute an
attention‑style weighting across patches, yielding an aggregated
feature vector per input sample.  The implementation uses Qiskit
and the Aer simulator."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import BaseBackend
from qiskit.providers.aer import AerSimulator


def SelfAttention():
    """Factory returning a :class:`HybridSelfAttention` instance."""
    return HybridSelfAttention(n_qubits=4, patch_size=2)


class HybridSelfAttention:
    """Hybrid quantum self‑attention with patch‑wise quantum kernels."""

    def __init__(self, n_qubits: int = 4, patch_size: int = 2):
        self.n_qubits = n_qubits
        self.patch_size = patch_size
        self.backend = AerSimulator()

    def _encode_patch(self, circuit: QuantumCircuit, patch: np.ndarray, rotation_params: np.ndarray):
        """Encode a 2×2 patch into ``n_qubits`` using rotation gates."""
        # Assume patch is flattened to length n_qubits
        for i, val in enumerate(patch):
            # Map pixel value to rotation angle
            circuit.ry(rotation_params[i] * val, i)

    def _build_patch_circuit(self, patch: np.ndarray, rotation_params: np.ndarray,
                             entangle_params: np.ndarray) -> QuantumCircuit:
        """Build a single‑patch variational circuit."""
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circ = QuantumCircuit(qr, cr)
        self._encode_patch(circ, patch, rotation_params)
        # Entanglement layer (CX gates)
        for i in range(self.n_qubits - 1):
            circ.cx(i, i + 1)
        # Additional entanglement with parameters
        for i, angle in enumerate(entangle_params):
            circ.rz(angle, i)
        circ.measure(qr, cr)
        return circ

    def _extract_patches(self, inputs: np.ndarray) -> np.ndarray:
        """Return a flattened patch array of shape (batch, n_patches, n_qubits)."""
        arr = inputs.astype(np.float32)
        if arr.ndim == 4:
            b, c, h, w = arr.shape
            kernel = self.patch_size
            patches = []
            for i in range(0, h, kernel):
                for j in range(0, w, kernel):
                    patch = arr[:, :, i:i+kernel, j:j+kernel]
                    patch = patch.reshape(b, -1)
                    patches.append(patch)
            patches = np.stack(patches, axis=1)  # (b, n_patches, d)
        else:
            # assume already flattened
            patches = arr.reshape(arr.shape[0], 1, -1)
        return patches

    def run(
        self,
        backend: BaseBackend | None = None,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
        inputs: np.ndarray | None = None,
        shots: int = 1024,
    ) -> np.ndarray:
        """Execute the hybrid quantum self‑attention.

        Parameters
        ----------
        backend : BaseBackend, optional
            Quantum backend (default Aer simulator).
        rotation_params : np.ndarray, optional
            Parameters for the rotation gates per qubit.
        entangle_params : np.ndarray, optional
            Parameters for the entanglement (rz) gates per qubit.
        inputs : np.ndarray
            Image or flattened patch data.
        shots : int
            Number of shots per circuit.

        Returns
        -------
        np.ndarray
            Aggregated feature vector of shape ``(batch, n_qubits)``.
        """
        if backend is None:
            backend = self.backend
        if rotation_params is None or entangle_params is None or inputs is None:
            raise ValueError("rotation_params, entangle_params and inputs must be provided.")
        patches = self._extract_patches(inputs)  # (b, n_patches, d)
        b, n_patches, d = patches.shape
        all_vectors = []
        for sample_idx in range(b):
            sample_vectors = []
            for patch_idx in range(n_patches):
                circ = self._build_patch_circuit(
                    patches[sample_idx, patch_idx], rotation_params, entangle_params)
                job = qiskit.execute(circ, backend, shots=shots)
                counts = job.result().get_counts(circ)
                vec = np.zeros(d)
                for state, cnt in counts.items():
                    bits = np.array([int(bit) for bit in state[::-1]])
                    vec += cnt * (1 - 2 * bits)  # +1 for |0>, -1 for |1>
                vec /= shots
                sample_vectors.append(vec)
            sample_vectors = np.stack(sample_vectors, axis=0)  # (n_patches, d)
            # Classical attention over patches
            scores = np.exp(np.sum(sample_vectors, axis=1))
            weights = scores / np.sum(scores)
            out = np.sum(sample_vectors * weights[:, None], axis=0)
            all_vectors.append(out)
        return np.stack(all_vectors, axis=0)


__all__ = ["SelfAttention", "HybridSelfAttention"]
