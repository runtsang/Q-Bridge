"""UnifiedHybridRegressor – quantum‑enhanced regressor with self‑attention.

This module extends the classical version by replacing the final linear
head with a parameterized quantum circuit.  The architecture follows
the same pattern: a 3‑layer MLP, a self‑attention block, and a quantum
layer that computes the expectation value of an observable.  The
quantum part is differentiable via a custom autograd function, enabling
end‑to‑end training with PyTorch optimizers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.providers import BaseBackend

class _SelfAttention(nn.Module):
    """Classical self‑attention inspired by SelfAttention.py."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.W_q(x)
        key   = self.W_k(x)
        value = self.W_v(x)
        scores = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.embed_dim ** 0.5), dim=-1)
        return torch.matmul(scores, value)

class QuantumCircuitWrapper:
    """Parameterized circuit executed on AerSimulator."""
    def __init__(self, n_qubits: int, backend: BaseBackend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = QiskitCircuit(n_qubits)
        for i in range(n_qubits):
            self.circuit.h(i)
            self.circuit.rx(self.theta, i)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            parameter_binds=[{self.theta: thetas[0]}],
            shots=self.shots,
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        angles = inputs.detach().cpu().numpy()
        exp_out = []
        for a in angles:
            exp_out.append(ctx.circuit.run([a])[0])
        out = torch.tensor(exp_out, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            right = ctx.circuit.run([val + shift])[0]
            left = ctx.circuit.run([val - shift])[0]
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class UnifiedHybridRegressor(nn.Module):
    """Quantum‑enhanced regressor with self‑attention."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8,
                 embed_dim: int = 4, n_qubits: int = 1,
                 backend: BaseBackend = AerSimulator(),
                 shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, embed_dim),
        )
        self.attention = _SelfAttention(embed_dim=embed_dim)
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        attn_out = self.attention(hidden)
        scalar = attn_out.mean(dim=1)
        return HybridFunction.apply(scalar.squeeze(), self.quantum_circuit, self.shift)

def EstimatorQNN() -> UnifiedHybridRegressor:
    return UnifiedHybridRegressor()

__all__ = ["UnifiedHybridRegressor", "EstimatorQNN"]
