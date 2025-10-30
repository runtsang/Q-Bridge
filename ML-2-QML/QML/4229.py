"""Quantum hybrid convolutional network with attention.

This module defines a drop‑in replacement for Conv.py that
- applies a classical convolution filter,
- feeds the embedded features into a quantum‑parameterized filter,
- runs a quantum self‑attention block, and
- produces a two‑class probability vector.

The implementation mirrors the classical ConvGen220 while replacing
the attention head with a variational quantum circuit.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit


class QuantumFilter:
    """Parametrised quantum filter that maps a classical vector to a probability."""
    def __init__(self, n_qubits: int, backend, shots: int = 1024, threshold: float = 0.5):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

        self.circuit = QuantumCircuit(n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Run the filter on a 1‑D array of length n_qubits."""
        flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in flat:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}
            param_binds.append(bind)
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


class QuantumSelfAttention:
    """Variational quantum self‑attention block."""
    def __init__(self, n_qubits: int, backend, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.qr = QuantumRegister(n_qubits)
        self.cr = ClassicalRegister(n_qubits)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> float:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=self.shots)
        result = job.result().get_counts(circuit)

        exp = 0
        for key, val in result.items():
            z = 1 - 2 * int(key, 2)
            exp += z * val
        return exp / self.shots


class ConvGen220(nn.Module):
    """Quantum‑augmented convolutional feature extractor with attention."""
    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        attention_dim: int = 4,
        dropout: float = 0.2,
        shots: int = 512,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.dropout = nn.Dropout(dropout)

        self.embed = None
        self.attention_dim = attention_dim

        self.rotation_params = nn.Parameter(
            torch.randn(attention_dim * 3), requires_grad=True
        )
        self.entangle_params = nn.Parameter(
            torch.randn(attention_dim - 1), requires_grad=True
        )

        self.backend = Aer.get_backend("qasm_simulator")
        self.quantum_filter = QuantumFilter(
            n_qubits=self.attention_dim,
            backend=self.backend,
            shots=shots,
            threshold=conv_threshold,
        )
        self.quantum_attention = QuantumSelfAttention(
            n_qubits=attention_dim,
            backend=self.backend,
            shots=shots,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, H, W)
        Returns:
            Tensor of shape (B, 2) containing class probabilities.
        """
        conv_out = torch.sigmoid(self.conv(x) - self.conv_threshold)

        B, C, H, W = conv_out.shape
        flat = conv_out.view(B, -1)
        if self.embed is None or self.embed.in_features!= flat.size(1):
            self.embed = nn.Linear(flat.size(1), self.attention_dim, bias=True)
            self.add_module("embed", self.embed)
        embedded = self.embed(flat)

        q_probs = []
        for i in range(B):
            vec = embedded[i].detach().cpu().numpy()
            q_probs.append(self.quantum_filter.run(vec))
        q_probs = np.array(q_probs)

        rot = self.rotation_params.detach().cpu().numpy()
        ent = self.entangle_params.detach().cpu().numpy()
        attn_vals = []
        for i in range(B):
            attn_vals.append(self.quantum_attention.run(rot, ent))
        attn_vals = np.array(attn_vals)

        combined = q_probs + attn_vals
        combined_tensor = torch.tensor(combined, dtype=torch.float32, device=x.device)
        prob = torch.sigmoid(combined_tensor).unsqueeze(1)
        return torch.cat((prob, 1 - prob), dim=1)


__all__ = ["ConvGen220"]
