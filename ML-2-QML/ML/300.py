"""Hybrid self‑attention combining classical transformer attention and a quantum‑enhanced block.

The class exposes a forward that returns:
- classical_attention: the output of a standard scaled dot‑product attention
- quantum_attention: a probability distribution obtained from a quantum circuit
- fused_attention: a weighted sum of the two, where the weight is a learnable gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class SelfAttentionDual(nn.Module):
    def __init__(self, embed_dim: int, gate_init: float = 0.5, n_qubits: int = 4):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.
        gate_init : float
            Initial value for the learnable gate (between 0 and 1).
        n_qubits : int
            Number of qubits used in the quantum attention block.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits

        # Classical attention parameters
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Learnable gate controlling the fusion of classical and quantum attentions
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def classical_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled dot‑product attention using linear projections.
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    def forward(self, x: torch.Tensor, quantum_attention: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        quantum_attention : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim) produced by the quantum block.

        Returns
        -------
        classical_out : torch.Tensor
            Output of the classical attention.
        quantum_out : torch.Tensor
            Output of the quantum attention.
        fused_out : torch.Tensor
            Weighted sum: gate * classical_out + (1 - gate) * quantum_attention.
        """
        classical_out = self.classical_attention(x)
        gate = torch.sigmoid(self.gate)  # keep gate in (0,1)
        fused_out = gate * classical_out + (1 - gate) * quantum_attention
        return classical_out, quantum_attention, fused_out
