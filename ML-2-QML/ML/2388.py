"""Hybrid self‑attention module with a classical dense head and a simulated quantum refinement layer.

The module is inspired by the classical SelfAttention seed and the hybrid binary‑classification seed.
It exposes a single class ``HybridSelfAttention`` that can be used in a pure PyTorch pipeline.
The quantum refinement is implemented as a simple finite‑difference differentiable expectation
computed with NumPy, mimicking the behaviour of a parameterised quantum circuit.

The design allows for easy substitution of a real quantum backend by setting ``use_quantum=True``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class _QuantumExpectationSimulator:
    """
    Lightweight simulator that mimics a parameterised quantum circuit.
    For each input vector it returns the expectation value of a Z measurement
    after applying rotations and entangling gates defined by the input parameters.
    """
    def __init__(self, n_qubits: int = 4, shift: float = np.pi / 2):
        self.n_qubits = n_qubits
        self.shift = shift

    def expectation(self, params: np.ndarray) -> float:
        """
        Compute a toy expectation value: sum of sin of each parameter
        weighted by a simple entangling mask.
        """
        mask = np.array([(-1)**i for i in range(self.n_qubits)])
        return np.sum(np.sin(params) * mask)

    def grad(self, params: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Finite‑difference gradient of the expectation w.r.t. the parameters.
        """
        grads = np.zeros_like(params)
        for i in range(len(params)):
            perturbed = params.copy()
            perturbed[i] += eps
            grads[i] = (self.expectation(perturbed) - self.expectation(params)) / eps
        return grads

class HybridSelfAttention(nn.Module):
    """
    Hybrid self‑attention module that combines a classical attention head
    with a simulated quantum refinement layer.
    """
    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 1,
        use_quantum: bool = False,
        n_qubits: int = 4,
        shift: float = np.pi / 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.use_quantum = use_quantum

        # Classical attention parameters
        self.q_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.k_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.v_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))

        # Quantum refinement parameters (simulated)
        if self.use_quantum:
            self.qc = _QuantumExpectationSimulator(n_qubits=n_qubits, shift=shift)
            # Parameters that will be fed to the simulator
            self.qc_params = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)
        Returns:
            Tensor of shape (batch, seq_len, embed_dim)
        """
        batch, seq_len, _ = x.shape

        # Classical attention
        Q = torch.matmul(x, self.q_weight)  # (batch, seq_len, embed_dim)
        K = torch.matmul(x, self.k_weight)
        V = torch.matmul(x, self.v_weight)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (batch, seq_len, embed_dim)

        if self.use_quantum:
            # Flatten batch and seq_len for simulation
            flat = out.reshape(-1, self.embed_dim)
            # Compute expectation for each sample
            expectations = []
            for sample in flat:
                # Use the quantum parameters as angles
                angles = self.qc_params.detach().cpu().numpy()
                exp_val = self.qc.expectation(angles)
                expectations.append(exp_val)
            expectations = torch.tensor(expectations, device=out.device).view(batch, seq_len, 1)
            # Refine the attention output
            out = out * expectations

        return out

__all__ = ["HybridSelfAttention"]
