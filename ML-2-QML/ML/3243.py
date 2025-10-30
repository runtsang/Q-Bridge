"""SelfAttentionHybrid – a hybrid classical‑quantum multi‑head attention module.

The module marries the classical multi‑head attention mechanism from the first
seed with a quantum expectation head inspired by the second seed.  The quantum
head is instantiated per selected head and refines each head’s attention
scores by multiplying them with a scalar weight produced by a variational
circuit.  The design preserves the classical efficiency of attention while
adding a learnable quantum contribution that can be trained end‑to‑end
with PyTorch autograd.

Dependencies
-------------
* torch
* numpy
* quantum_attention (see qml_code below)
* qiskit (indirectly via quantum_attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import quantum_attention as qa


class SelfAttentionHybrid(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, quantum_heads: int = 1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.quantum_heads = min(quantum_heads, num_heads)

        # Classical linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Quantum refinement layers for the first `quantum_heads` heads
        backend = qa.qiskit.Aer.get_backend("aer_simulator")
        self.quantum_layers = nn.ModuleList(
            [qa.Hybrid(self.head_dim, backend, shots=256, shift=np.pi / 2)
             for _ in range(self.quantum_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        batch, seq_len, _ = x.size()

        # Linear projections
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        # Quantum refinement on selected heads
        if self.quantum_heads > 0:
            # Aggregate each head’s attention into a scalar (mean over seq_len × seq_len)
            agg = attn[:, :self.quantum_heads].mean(dim=(2, 3))  # shape (batch, quantum_heads)
            # Quantum head produces a scalar weight per head per batch
            refined = torch.stack(
                [self.quantum_layers[i](agg[:, i].unsqueeze(1))
                 for i in range(self.quantum_heads)],
                dim=1,
            )  # shape (batch, quantum_heads, 1)
            refined = refined.view(batch, self.quantum_heads, 1, 1)  # broadcastable
            # Apply refined weights
            attn[:, :self.quantum_heads] = attn[:, :self.quantum_heads] * refined

        # Weighted sum of values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return out


__all__ = ["SelfAttentionHybrid"]
