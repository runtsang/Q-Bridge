"""Hybrid classical‑quantum convolution‑attention network for binary classification.

The module unifies:
* A learnable convolution filter (from Conv.py)
* A self‑attention module (from SelfAttention.py)
* A hybrid dense head with a differentiable quantum expectation (from ClassicalQuantumBinaryClassification.py)
* Optional quantum enhancements for the attention block (from QTransformerTorch.py)
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit

# --------------------------------------------------------------------------- #
# 1. Classical convolution filter – learnable kernel with threshold
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Learnable 2×2 convolution followed by a sigmoid threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        return torch.sigmoid(logits - self.threshold)

# --------------------------------------------------------------------------- #
# 2. Hybrid self‑attention – classical backbone + optional quantum head
# --------------------------------------------------------------------------- #
class SelfAttentionHybrid(nn.Module):
    """Wraps a classical Multi‑Head Attention with an optional quantum layer."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 q_device=None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_quantum = use_quantum
        self.q_device = q_device

        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

        if use_quantum:
            self.q_layer = nn.ModuleList(
                [nn.Linear(1, 1, bias=False) for _ in range(num_heads)]
            )
            self.theta = nn.Parameter(torch.randn(num_heads))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        if self.use_quantum:
            B, T, E = x.shape
            heads = x.unbind(dim=1)
            refined = []
            for h, head in enumerate(heads):
                qubit = self.q_layer[h](head)
                qubit = torch.cos(self.theta[h]) * qubit + torch.sin(self.theta[h]) * (1 - qubit)
                refined.append(qubit)
            return torch.stack(refined, dim=1)
        return attn_out

# --------------------------------------------------------------------------- #
# 3. Hybrid dense head – classical linear + differentiable quantum expectation
# --------------------------------------------------------------------------- #
class HybridHead(nn.Module):
    """Dense layer followed by a quantum expectation head."""
    def __init__(self,
                 in_features: int,
                 n_qubits: int = 2,
                 backend=None,
                 shots: int = 1024,
                 shift: float = np.pi / 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, n_qubits)
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.shift = shift

        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.theta_params = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i, theta in enumerate(self.theta_params):
            self.circuit.rx(theta, i)
        self.circuit.measure_all()

    def _expectation(self, counts: dict[str, int]) -> float:
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        signs = (-1) ** states
        return float(np.sum(signs * probs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin = self.linear(x)
        angles = lin.detach().cpu().numpy()
        binds = [{self.theta_params[i]: angles[b, i] for i in range(self.n_qubits)} for b in range(angles.shape[0])]
        compiled = qiskit.transpile(self.circuit, self.backend)
        qobj = qiskit.assemble(compiled,
                               shots=self.shots,
                               parameter_binds=binds)
        job = self.backend.run(qobj)
        counts_list = job.result().get_counts(self.circuit)
        exp_vals = [self._expectation(counts) for counts in counts_list]
        return torch.tensor(exp_vals, device=x.device, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# 4. Full hybrid network – convolution → attention → hybrid head
# --------------------------------------------------------------------------- #
class ConvGenHybrid(nn.Module):
    """End‑to‑end binary classifier that stitches together the three building blocks."""
    def __init__(self,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 n_qubits: int = 2,
                 use_quantum_attn: bool = False,
                 q_device=None) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=2, threshold=0.1)
        self.attn = SelfAttentionHybrid(embed_dim, num_heads,
                                        dropout=0.1,
                                        use_quantum=use_quantum_attn,
                                        q_device=q_device)
        self.proj = nn.Linear(1, embed_dim)
        self.head = HybridHead(in_features=embed_dim,
                               n_qubits=n_qubits,
                               backend=qiskit.Aer.get_backend("qasm_simulator"),
                               shots=512,
                               shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)               # (B, 1, H-1, W-1)
        seq = conv_out.view(x.size(0), -1, conv_out.size(1))  # (B, S, 1)
        proj = self.proj(seq)                # (B, S, embed_dim)
        attn_out = self.attn(proj, mask=None)  # (B, S, embed_dim)
        attn_mean = attn_out.mean(dim=1)      # (B, embed_dim)
        out = self.head(attn_mean)             # (B,)
        return torch.sigmoid(out).unsqueeze(-1)

__all__ = ["ConvGenHybrid"]
