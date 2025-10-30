"""
Module: QuantumConvAttention
Author: gpt-oss-20b
Description:
    Quantum analogue of ConvAttentionModule. Each image patch is processed by a
    parameterised quantum convolution circuit (QuanvCircuit). The resulting
    probabilities are assembled into a feature vector and passed to a quantum
    self‑attention circuit (QuantumSelfAttention). The final measurement
    counts provide a scalar output.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

# Import the seed quantum circuits
from Conv import Conv as QuantumConv
from SelfAttention import SelfAttention as QuantumSelfAttention


class QuantumConvAttention:
    """
    Quantum convolution + self‑attention pipeline.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolutional kernel (patch size).
    threshold : float, default 127
        Threshold used by the quantum convolution.
    embed_dim : int, default 4
        Dimensionality of the attention embedding.
    shots : int, default 1024
        Number of shots for each circuit execution.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 127, embed_dim: int = 4, shots: int = 1024) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.embed_dim = embed_dim
        self.shots = shots

        # Backend for all quantum circuits
        self.backend = Aer.get_backend("qasm_simulator")

        # Quantum convolution circuit
        self.conv_circuit = QuantumConv(kernel_size=kernel_size, backend=self.backend,
                                        shots=shots, threshold=threshold)

        # Quantum self‑attention circuit
        self.attention = QuantumSelfAttention(n_qubits=embed_dim)

    def _patch_features(self, patch: np.ndarray) -> float:
        """
        Run the quantum convolution on a single patch and return the average
        probability of measuring |1> across all qubits.

        Parameters
        ----------
        patch : np.ndarray
            2‑D array of shape (k, k) with integer pixel values.

        Returns
        -------
        prob : float
            Average probability of |1>.
        """
        return self.conv_circuit.run(patch)

    def run(self, image: np.ndarray) -> float:
        """
        Execute the full pipeline on a single image.

        Parameters
        ----------
        image : np.ndarray
            2‑D array of shape (H, W) with integer pixel values.

        Returns
        -------
        output : float
            Scalar result of the quantum self‑attention block.
        """
        H, W = image.shape
        patches = []
        # 1. Extract patches
        for i in range(H - self.kernel_size + 1):
            for j in range(W - self.kernel_size + 1):
                patch = image[i:i+self.kernel_size, j:j+self.kernel_size]
                patches.append(patch)

        # 2. Quantum convolution per patch
        patch_probs = np.array([self._patch_features(p) for p in patches])

        # 3. Build attention parameter arrays from the probabilities
        # For simplicity, use the probabilities as rotation angles
        rotation_params = patch_probs.repeat(self.embed_dim).astype(np.float32)
        entangle_params = np.zeros(self.embed_dim - 1, dtype=np.float32)

        # 4. Quantum self‑attention
        counts = self.attention.run(self.backend, rotation_params, entangle_params, shots=self.shots)

        # 5. Convert measurement counts to a scalar (average |1> proportion)
        total_counts = sum(counts.values())
        if total_counts == 0:
            return 0.0
        ones = sum(sum(int(bit) for bit in key) * val for key, val in counts.items())
        return ones / (total_counts * self.embed_dim)


__all__ = ["QuantumConvAttention"]
