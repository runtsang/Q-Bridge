"""Quantum‑classical hybrid self‑attention module.

This module combines the dense‑attention logic from the classical seed
(`SelfAttention.py`) with a Qiskit variational circuit that mirrors the
quantum seed.  It also incorporates the parameter‑clipping strategy
used in the fraud‑detection example to keep rotation and entanglement
angles bounded during training.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

# --------------------------------------------------------------------------- #
#  Classical dense‑attention sub‑module
# --------------------------------------------------------------------------- #
class _DenseAttention(nn.Module):
    """Linear projection of query/key/value with optional clipping."""
    def __init__(self, embed_dim: int, hidden_dim: int = 32, clip: bool = True):
        super().__init__()
        self.clip = clip
        self.query = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.key   = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        # Apply clipping to the provided parameters before using them
        if self.clip:
            rotation_params = np.clip(rotation_params, -np.pi, np.pi)
            entangle_params = np.clip(entangle_params, -np.pi, np.pi)

        # Convert numpy params to tensors for weight assignment
        with torch.no_grad():
            self.query.weight.copy_(torch.tensor(rotation_params.reshape(self.query.out_features, -1)))
            self.key.weight.copy_(torch.tensor(entangle_params.reshape(self.key.out_features, -1)))

        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        scores = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(q.size(-1)), dim=-1)
        return scores @ v


# --------------------------------------------------------------------------- #
#  Quantum attention sub‑module (Qiskit)
# --------------------------------------------------------------------------- #
class _QuantumAttention(nn.Module):
    """Variational circuit that outputs a probability distribution over qubits."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self._build_circuit()

    def _build_circuit(self):
        import qiskit
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

        self.qr = QuantumRegister(self.n_qubits, "q")
        self.cr = ClassicalRegister(self.n_qubits, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)

        # Parameter placeholders
        self.rotation_params = [self.circuit.rx(0, i) for i in range(self.n_qubits)]
        self.entangle_params = [self.circuit.cx(i, i+1) for i in range(self.n_qubits-1)]

        self.circuit.measure(self.qr, self.cr)

        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def forward(self, rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                shots: int = 1024) -> torch.Tensor:
        import qiskit
        from qiskit import execute

        # Clip parameters to a reasonable range
        rotation_params = np.clip(rotation_params, -np.pi, np.pi)
        entangle_params = np.clip(entangle_params, -np.pi, np.pi)

        # Update circuit with new parameters
        for i, val in enumerate(rotation_params):
            self.circuit.data[i].operation.params = [val]
        for i, val in enumerate(entangle_params):
            self.circuit.data[self.n_qubits + i].operation.params = [val]

        job = execute(self.circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Convert counts to a probability distribution
        probs = np.zeros(self.n_qubits)
        for state, cnt in counts.items():
            idx = int(state, 2)
            probs[idx] = cnt / shots
        return torch.tensor(probs, dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  Hybrid self‑attention module
# --------------------------------------------------------------------------- #
class QuantumSelfAttentionHybrid(nn.Module):
    """Combines classical dense attention with a quantum attention head."""
    def __init__(self, embed_dim: int, hidden_dim: int = 32, n_qubits: int = 4):
        super().__init__()
        self.classical = _DenseAttention(embed_dim, hidden_dim)
        self.quantum = _QuantumAttention(n_qubits)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: torch.Tensor,
            shots: int = 1024) -> torch.Tensor:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits * 3,) – rotation angles for each qubit.
        entangle_params : np.ndarray
            Shape (n_qubits - 1,) – entanglement angles.
        inputs : torch.Tensor
            Input batch of shape (batch, embed_dim).
        shots : int, optional
            Number of shots for quantum simulation.

        Returns
        -------
        torch.Tensor
            Attention-weighted representation of size (batch, embed_dim).
        """
        # Classical attention
        classical_out = self.classical(inputs, rotation_params, entangle_params)

        # Quantum attention distribution
        quantum_dist = self.quantum(rotation_params, entangle_params, shots)

        # Broadcast quantum distribution across batch
        quantum_dist = quantum_dist.unsqueeze(0).expand_as(classical_out)

        # Fuse: weighted sum of classical output and quantum distribution
        return classical_out * quantum_dist.unsqueeze(-1)

__all__ = ["QuantumSelfAttentionHybrid"]
