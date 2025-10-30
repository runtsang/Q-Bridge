"""Hybrid self‑attention module combining classical RBF kernel attention with variational encoding.

The class inherits from the original SelfAttention interface but augments the
similarity computation with a kernel‑based similarity measure. It uses the
Kernel class from QuantumKernelMethod (classical RBF) to compute pairwise
similarities between query and key vectors. The rotation and entangle
parameters are used to linearly transform the inputs before kernel evaluation.
"""

import numpy as np
import torch
from.QuantumKernelMethod import Kernel

class HybridSelfAttention:
    """Hybrid self‑attention with kernel‑based similarity."""
    def __init__(self, embed_dim: int = 4, gamma: float = 1.0):
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.kernel = Kernel(gamma)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # Linear transformation of inputs
        rot_mat = rotation_params.reshape(self.embed_dim, -1)
        ent_mat = entangle_params.reshape(self.embed_dim, -1)
        query = torch.as_tensor(inputs @ rot_mat.T, dtype=torch.float32)
        key   = torch.as_tensor(inputs @ ent_mat.T, dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)

        # Compute pairwise RBF kernel similarity
        diff = query.unsqueeze(1) - key.unsqueeze(0)
        sq_norm = torch.sum(diff * diff, dim=-1)
        scores = torch.exp(-self.gamma * sq_norm)

        # Softmax to obtain attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn_weights, value)
        return out.numpy()

__all__ = ["HybridSelfAttention"]
