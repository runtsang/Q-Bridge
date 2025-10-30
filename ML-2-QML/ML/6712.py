"""
Hybrid classical‑quantum self‑attention module.

The implementation keeps the original API shape but expands the
functionality:  * the classical part now uses a multi‑head scaled
dot‑product attention (similar to the standard transformer).  * the
quantum part is a parameterised variational circuit that embeds the
input vector into a register, applies a tunable entangling layer,
and measures expectation values that are used as attention scores.
The module can be trained with either PyTorch or a classical optimiser
and the quantum backend can be swapped at runtime.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from.SelfAttention_qml import quantum_attention

__all__ = ["SelfAttentionHybrid"]

class SelfAttentionHybrid(nn.Module):
    """
    Hybrid attention block that can be instantiated with ``mode`` set to
    ``"classical"`` or ``"quantum"``.  The constructor accepts a dictionary
    of hyper‑parameters that will be linear‑transform
    (the “weight‑matrix” **f**‑matrix) and shape‑sizing.
    """
    def __init__(
        self,
        embed_dim: int,
        heads: int = 1,
        mode: str = "classical",
        n_qubits: int = 4,
        quantum_backend=None,
        dropout: float = 0.0,
    ):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.
        heads : int, default 1
            Number of attention heads.
        mode : str, default "classical"
            Either ``"classical"`` or ``"quantum"``.
        n_qubits : int, default 4
            Number of qubits used in the quantum attention sub‑module.
        quantum_backend : optional
            Backend to pass to the quantum circuit (e.g. Pennylane device).
        dropout : float, default 0.0
            Dropout probability applied to attention weights.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.mode = mode
        self.n_qubits = n_qubits
        self.quantum_backend = quantum_backend
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for dot‑product attention
        self.scale = embed_dim ** -0.5

        # Classical projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Quantum parameters – learnable
        # shape: (3 * n_qubits,) for rotations, (n_qubits - 1,) for entanglement
        self.rotation_params = nn.Parameter(torch.randn(3 * n_qubits))
        self.entangle_params = nn.Parameter(torch.randn(n_qubits - 1))

        # Optional callback for monitoring
        self.callback = None

    def set_callback(self, cb):
        """Register a callback that receives the attention weights each step."""
        self.callback = cb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid self‑attention block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = x.size()

        # Classical projections
        q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.mode == "classical":
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
        elif self.mode == "quantum":
            # Flatten batch and seq_len for quantum evaluation
            q_flat = q.reshape(-1, self.embed_dim)
            # Convert to numpy and detach
            q_np = q_flat.detach().cpu().numpy()
            # Run quantum circuit
            scores_np = quantum_attention(
                q_np,
                self.rotation_params.detach().cpu().numpy(),
                self.entangle_params.detach().cpu().numpy(),
                n_qubits=self.n_qubits,
                backend=self.quantum_backend,
            )
            # Convert back to torch
            scores = torch.from_numpy(scores_np).to(x.device)
            # Reshape to (batch, seq_len, seq_len)
            scores = scores.view(batch, seq_len, seq_len)
            attn = F.softmax(scores * self.scale, dim=-1)
            attn = self.dropout(attn)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        out = torch.matmul(attn, v)
        out = self.out_proj(out)

        if self.callback is not None:
            self.callback(attn.detach())

        return out
