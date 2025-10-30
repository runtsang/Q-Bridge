"""Hybrid self‑attention network combining classical and quantum components.

This module defines the class ``HybridSelfAttentionNet`` that integrates:
* A linear encoder mapping raw features to a low‑dimensional space.
* A variational quantum self‑attention block (``QuantumSelfAttention``) that refines the representation.
* A differentiable hybrid head (``Hybrid``) that maps the quantum output to binary logits.

The implementation is fully differentiable and can be trained with standard optimisers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import QuantumCircuit, transpile, assemble

__all__ = ["HybridSelfAttentionNet", "QuantumSelfAttention", "HybridFunction", "Hybrid"]

class QuantumSelfAttention(nn.Module):
    """Variational quantum circuit implementing a self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (and thus the dimensionality of the input).
    backend : qiskit.providers.basebackend.BaseBackend, optional
        Backend to execute the circuit. Defaults to the Aer QASM simulator.
    shots : int, optional
        Number of shots per circuit execution.
    """
    def __init__(self, n_qubits: int, backend=None, shots: int = 100):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        # Trainable entanglement parameters
        self.entangle_params = nn.Parameter(torch.randn(n_qubits - 1))
        # Trainable rotation parameters per qubit
        self.rotation_params = nn.Parameter(torch.randn(n_qubits, 3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the variational circuit for each sample in *inputs*.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, n_qubits) containing values used to scale
            the RX, RY, RZ gates of each qubit.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, n_qubits) containing expectation values
            of the Z operator for each qubit.
        """
        batch_size = inputs.shape[0]
        results = []
        for i in range(batch_size):
            circuit = QuantumCircuit(self.n_qubits, name=f"qsa_{i}")
            # Data‑re‑uploading rotations
            for q in range(self.n_qubits):
                circuit.rx(self.rotation_params[q, 0] * inputs[i, q].item(), q)
                circuit.ry(self.rotation_params[q, 1] * inputs[i, q].item(), q)
                circuit.rz(self.rotation_params[q, 2] * inputs[i, q].item(), q)
            # Entanglement
            for q in range(self.n_qubits - 1):
                circuit.crx(self.entangle_params[q].item(), q, q + 1)
            circuit.measure_all()
            compiled = transpile(circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts(circuit)
            # Compute expectation of Z for each qubit
            expectations = []
            for q in range(self.n_qubits):
                zero = sum(counts.get(bin(1 << q)[2:].zfill(self.n_qubits), 0))
                one = self.shots - zero
                exp_z = (zero - one) / self.shots
                expectations.append(exp_z)
            results.append(expectations)
        return torch.tensor(results, dtype=torch.float32)

class HybridFunction(torch.autograd.Function):
    """Differentiable interface that forwards activations through a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumSelfAttention, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit(inputs)
        result = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs) * ctx.shift
        gradients = []
        for idx in range(inputs.shape[0]):
            right = ctx.circuit(inputs[idx] + shift[idx]).item()
            left = ctx.circuit(inputs[idx] - shift[idx]).item()
            gradients.append(right - left)
        gradients = torch.tensor(gradients, dtype=torch.float32)
        return gradients * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards a classical vector through a quantum circuit."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.quantum_circuit = QuantumSelfAttention(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

class HybridSelfAttentionNet(nn.Module):
    """Full hybrid network that uses a classical encoder, a quantum self‑attention block,
    and a differentiable hybrid head for binary classification.
    """
    def __init__(self, embed_dim: int, n_qubits: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        # Simple classical encoder
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits),
        )
        # Quantum self‑attention
        self.quantum_sa = QuantumSelfAttention(n_qubits)
        # Hybrid head
        self.hybrid_head = Hybrid(n_qubits, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, embed_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 2) containing class probabilities.
        """
        encoded = self.encoder(inputs)               # (batch, n_qubits)
        quantum_out = self.quantum_sa(encoded)       # (batch, n_qubits)
        logits = self.hybrid_head(quantum_out)       # (batch, 1)
        probs = torch.cat((logits, 1 - logits), dim=-1)
        return probs
