"""Quantum self‑attention module built with Pennylane.

The module encodes CNN features into a qubit register, applies
parameterised rotations and CRX entangling gates, then measures the
expectation of Pauli‑Z on each qubit. The resulting expectation values
are interpreted as attention logits, soft‑maxed, and used to weight the
original feature vector. The public interface mirrors the classical
variant for easy backend swapping."""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn.functional as F

class SelfAttentionHybridQML:
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.device = qml.device("default.qubit", wires=n_qubits)
        # QNode that returns a list of Pauli‑Z expectation values
        self._circuit = qml.QNode(self._build_circuit, self.device, interface="numpy")

    def _build_circuit(self,
                       angles: np.ndarray,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> list[float]:
        # Angle encoding of input features
        for i, a in enumerate(angles):
            qml.RY(a, wires=i)
        # Parameterised single‑qubit rotations
        for i in range(self.n_qubits):
            qml.RX(rotation_params[3 * i], wires=i)
            qml.RY(rotation_params[3 * i + 1], wires=i)
            qml.RZ(rotation_params[3 * i + 2], wires=i)
        # CRX entangling layer
        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Length‑3*n_qubits rotation parameters.
        entangle_params : np.ndarray
            Length‑(n_qubits-1) entanglement parameters.
        inputs : np.ndarray
            Batch of grayscale images, shape (B,1,28,28).
        shots : int
            Number of shots for the backend (ignored for the default simulator).

        Returns
        -------
        np.ndarray
            Attention‑weighted embedding of shape (B, n_qubits).
        """
        # Classical CNN feature extraction to reuse the same embedding space
        x = torch.as_tensor(inputs, dtype=torch.float32)
        feats = F.max_pool2d(x, kernel_size=6).view(x.shape[0], -1)  # (B, 784)
        # Normalise features to the range [-π, π] for angle encoding
        angles = feats.numpy()
        amin, amax = angles.min(), angles.max()
        angles = 2 * np.pi * (angles - amin) / (amax - amin + 1e-8)
        # Run the quantum circuit for each sample
        outputs = []
        for a in angles:
            expvals = self._circuit(a, rotation_params, entangle_params)
            # Convert expectation values to attention logits
            logits = np.array(expvals)
            attn = np.exp(logits) / np.sum(np.exp(logits))
            # Weighted sum of the original feature vector
            weighted = attn @ a
            outputs.append(weighted)
        return np.array(outputs)

def SelfAttentionHybrid() -> SelfAttentionHybridQML:
    """Return a fully initialised quantum hybrid attention module."""
    return SelfAttentionHybridQML()

__all__ = ["SelfAttentionHybrid"]
