"""HybridSelfAttention: quantum‑enhanced self‑attention module with optional LSTM gate, fully‑connected quantum head, and quanvolution preprocessing."""

from __future__ import annotations

import math
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.providers.aer import AerSimulator
import torch
import torch.nn as nn
from typing import Optional


class HybridSelfAttention:
    """
    Quantum self‑attention module that mirrors the classical API.
    It supports:
      * encoding of input vectors into qubit rotations
      * optional entanglement via CRX gates
      * optional quantum LSTM‑style gating (simple 2‑qubit circuit)
      * a quantum fully‑connected head (parameterized circuit)
      * quantum convolution on image patches
    """

    def __init__(
        self,
        n_qubits: int,
        use_lstm: bool = False,
        use_fc: bool = True,
        use_conv: bool = False,
    ) -> None:
        self.n_qubits = n_qubits
        self.use_lstm = use_lstm
        self.use_fc = use_fc
        self.use_conv = use_conv

        self.backend = AerSimulator()
        self.shots = 1024

        # Quantum LSTM gate
        if self.use_lstm:
            self.lstm_circuit = self._build_lstm_circuit()

        # Quantum fully‑connected head
        if self.use_fc:
            self.fc_circuit = self._build_fc_circuit()

        # Quantum convolution filter
        if self.use_conv:
            self.conv_circuit = self._build_conv_circuit()

    def _build_lstm_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(np.pi / 4, i)
        return qc

    def _build_fc_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(np.pi / 3, i)
        return qc

    def _build_conv_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Simple entanglement pattern for a 2x2 patch
        qc.crx(np.pi / 2, 0, 1)
        qc.crx(np.pi / 2, 1, 2)
        qc.crx(np.pi / 2, 2, 3)
        return qc

    def _encode_input(self, qc: QuantumCircuit, vector: np.ndarray, rotation_params: np.ndarray) -> None:
        """Encode a classical vector into qubit rotations."""
        # Assume vector length equals n_qubits
        for i, val in enumerate(vector):
            qc.ry(val, i)

    def _measure_expectations(self, qc: QuantumCircuit) -> np.ndarray:
        """Return expectation values of Pauli‑Z for each qubit."""
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(qc)
        probs = np.zeros(2 ** self.n_qubits)
        for state, cnt in counts.items():
            probs[int(state, 2)] = cnt / self.shots
        expectations = np.zeros(self.n_qubits)
        for idx, p in enumerate(probs):
            bits = bin(idx)[2:].zfill(self.n_qubits)
            for qubit, bit in enumerate(reversed(bits)):
                expectations[qubit] += p * (-1) ** int(bit)
        return expectations

    def run(
        self,
        rotation_params: Optional[np.ndarray] = None,
        entangle_params: Optional[np.ndarray] = None,
        inputs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Execute quantum self‑attention on ``inputs``.
        ``rotation_params``: array of shape (n_qubits,) for RX rotations.
        ``entangle_params``: array of shape (n_qubits-1,) for CRX angles.
        ``inputs``: 2‑D array (batch, embed_dim) or 4‑D array (batch, 1, 28, 28) for images.
        Returns a NumPy array of attended representations.
        """
        if inputs is None:
            raise ValueError("``inputs`` must be provided.")
        if rotation_params is None:
            rotation_params = np.zeros(self.n_qubits)
        if entangle_params is None:
            entangle_params = np.zeros(self.n_qubits - 1)

        # Handle image convolution
        if self.use_conv and inputs.ndim == 4:
            batch_size = inputs.shape[0]
            conv_features = []
            for b in range(batch_size):
                img = inputs[b, 0, :, :]
                patches = []
                for r in range(0, 28, 2):
                    for c in range(0, 28, 2):
                        patch = img[r : r + 2, c : c + 2].flatten()
                        qc = QuantumCircuit(self.n_qubits)
                        self._encode_input(qc, patch, rotation_params)
                        qc += self.conv_circuit
                        qc.measure_all()
                        exp = self._measure_expectations(qc)
                        patches.append(exp)
                conv_features.append(np.concatenate(patches))
            return np.stack(conv_features)

        # Sequence attention
        batch_size, seq_len, embed_dim = inputs.shape
        outputs = []
        for b in range(batch_size):
            seq_out = []
            for t in range(seq_len):
                vec = inputs[b, t, :]
                qc = QuantumCircuit(self.n_qubits)
                self._encode_input(qc, vec, rotation_params)
                for i, angle in enumerate(entangle_params):
                    qc.crx(angle, i, i + 1)
                if self.use_lstm:
                    qc += self.lstm_circuit
                if self.use_fc:
                    qc += self.fc_circuit
                qc.measure_all()
                exp = self._measure_expectations(qc)
                # Convert expectation vector into attention weight
                weight = np.tanh(np.mean(exp))
                seq_out.append(weight * vec)
            outputs.append(np.stack(seq_out))
        return np.stack(outputs)

def SelfAttention(
    n_qubits: int = 4,
    use_lstm: bool = False,
    use_fc: bool = True,
    use_conv: bool = False,
) -> HybridSelfAttention:
    """Factory returning a quantum hybrid self‑attention instance."""
    return HybridSelfAttention(n_qubits, use_lstm, use_fc, use_conv)


__all__ = ["HybridSelfAttention", "SelfAttention"]
