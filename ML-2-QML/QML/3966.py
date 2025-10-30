"""Quantum self‑attention module that mirrors the classical branch.
It encodes each image patch into a 4‑qubit register, applies a rotation
layer derived from ``rotation_params`` and an entanglement layer from
``entangle_params``.  The expectation value of Pauli‑Z on each qubit
provides an attention score that is used to weight the classical patch
values.  The output shape matches that of the classical implementation
to enable seamless unit tests and downstream usage.

The construction deliberately uses the Aer state‑vector simulator
to keep the code lightweight and avoid external hardware dependencies.
"""

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.circuit.library.standard_gates import RY, CRX
import torch


class HybridSelfAttention:
    """
    Quantum‑enhanced self‑attention block.
    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits per patch (must match the patch size squared).
    """

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("statevector_simulator")

    def _extract_patches(self, inputs: np.ndarray) -> np.ndarray:
        """
        Convert a batch of grayscale images into non‑overlapping 2×2 patches.
        Returns an array of shape (batch, seq_len, 4).
        """
        batch, _, h, w = inputs.shape
        patch_size = 2
        seq_len = (h // patch_size) * (w // patch_size)
        patches = np.zeros((batch, seq_len, 4), dtype=np.float32)

        idx = 0
        for r in range(0, h, patch_size):
            for c in range(0, w, patch_size):
                patch = inputs[:, :, r : r + patch_size, c : c + patch_size]
                # Flatten 2×2 patch into 4‑dim vector
                patches[:, idx] = patch.reshape(batch, 4)
                idx += 1
        return patches

    def _circuit_for_patch(self, patch: np.ndarray, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Build a circuit that encodes a single 4‑dim patch.
        """
        qc = QuantumCircuit(self.n_qubits)
        # Encode each element of the patch into a Ry rotation
        for i in range(self.n_qubits):
            angle = patch[i] * rotation_params[i, i]  # use diagonal of rotation matrix
            qc.append(RY(angle), [i])
        # Entangle neighboring qubits
        for i in range(self.n_qubits - 1):
            qc.append(CRX(entangle_params[i]), [i, i + 1])
        return qc

    def _expectation_z(self, state: Statevector) -> float:
        """
        Compute the average Pauli‑Z expectation over all qubits.
        """
        probs = np.abs(state.data) ** 2
        exp = 0.0
        for idx, p in enumerate(probs):
            bits = format(idx, f"0{self.n_qubits}b")
            exp += p * np.prod([(-1) ** int(b) for b in bits])
        return exp

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
        backend=None,
    ) -> np.ndarray:
        """
        Compute attention‑weighted features using quantum evaluations.

        Parameters
        ----------
        rotation_params : np.ndarray
            4×4 rotation matrix (diagonal elements used).
        entangle_params : np.ndarray
            Array of 3 entanglement angles.
        inputs : np.ndarray
            Batch of grayscale images (batch, 1, H, W).
        shots : int
            Number of measurement shots (ignored when using statevector).
        backend : qiskit backend, optional
            Quantum backend; defaults to Aer statevector for deterministic output.

        Returns
        -------
        np.ndarray
            Attention‑weighted feature map of shape (batch, 4).
        """
        if backend is None:
            backend = self.backend

        patches = self._extract_patches(inputs)  # (batch, seq_len, 4)
        batch, seq_len, _ = patches.shape

        # Compute attention scores per patch via quantum expectation
        scores = np.zeros((batch, seq_len))
        for b in range(batch):
            for s in range(seq_len):
                qc = self._circuit_for_patch(patches[b, s], rotation_params, entangle_params)
                result = execute(qc, backend).result()
                state = Statevector.from_instruction(result.get_statevector(qc))
                exp = self._expectation_z(state)
                # Map expectation from [-1,1] to [0,1] for a positive attention weight
                scores[b, s] = (exp + 1) / 2

        # Normalize scores across patches per sample (softmax)
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        attn = e / e.sum(axis=1, keepdims=True)

        # Weighted sum of patch values
        weighted = np.sum(attn[:, :, None] * patches, axis=1)
        return weighted

__all__ = ["HybridSelfAttention"]
