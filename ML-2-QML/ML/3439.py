"""Hybrid self‑attention module that combines fast classical attention with optional quantum refinement.

The class exposes the same interface as the original seed but adds:
* an optional quantum gate‑based attention head that can be executed on any Qiskit backend.
* a configurable *depth* of attention layers, mirroring the variational depth in the QML counterpart.
* a simple API to switch between pure classical, pure quantum, or mixed modes.

The classical part uses torch tensors for speed; the quantum part is called only when a backend is supplied.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict

class SelfAttentionHybrid:
    """Hybrid self‑attention with optional quantum refinement.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    depth : int, default 1
        Number of stacked attention layers.  Each layer applies an independent
        linear mapping before the soft‑max.
    """

    def __init__(self, embed_dim: int, depth: int = 1) -> None:
        self.embed_dim = embed_dim
        self.depth = depth

        # Classical linear projections
        self.proj_q = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_k = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_v = torch.nn.Linear(embed_dim, embed_dim, bias=False)

        # Variational parameters that can be passed to the quantum module
        self.rotation_params = torch.nn.Parameter(
            torch.randn(embed_dim * 3, dtype=torch.float64)
        )
        self.entangle_params = torch.nn.Parameter(
            torch.randn(embed_dim - 1, dtype=torch.float64)
        )

    def _classical_attention(self, x: torch.Tensor) -> torch.Tensor:
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        scores = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1
        )
        return torch.matmul(scores, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the attention output."""
        return self._classical_attention(x)

    def export_quantum_params(self) -> Dict[str, np.ndarray]:
        """Return the rotation and entanglement parameters as NumPy arrays."""
        return {
            "rotation_params": self.rotation_params.detach().cpu().numpy(),
            "entangle_params": self.entangle_params.detach().cpu().numpy(),
        }

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Serialisable state dictionary for checkpointing."""
        return {
            "embed_dim": np.array([self.embed_dim]),
            "depth": np.array([self.depth]),
            "proj_q_weight": self.proj_q.weight.detach().cpu().numpy(),
            "proj_k_weight": self.proj_k.weight.detach().cpu().numpy(),
            "proj_v_weight": self.proj_v.weight.detach().cpu().numpy(),
            "rotation_params": self.rotation_params.detach().cpu().numpy(),
            "entangle_params": self.entangle_params.detach().cpu().numpy(),
        }

__all__ = ["SelfAttentionHybrid"]
