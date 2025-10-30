"""Classical self‑attention module with a linear post‑processing layer and parameter clipping.

The implementation follows the structure of the original SelfAttention seed but adds a
trainable scaling/shift layer inspired by the fraud‑detection example.  The attention
weights are computed from rotation and entanglement parameters, then passed through a
Tanh activation and a learnable affine transform.  All parameters are clipped to
[-clip_bounds, clip_bounds] to keep the optimisation stable.
"""

import numpy as np
import torch
from torch import nn

__all__ = ["SelfAttentionHybrid"]

class SelfAttentionHybrid:
    """Classical self‑attention with optional parameter clipping and affine post‑processing."""
    def __init__(self, embed_dim: int, clip_bounds: float = 5.0) -> None:
        self.embed_dim = embed_dim
        self.clip_bounds = clip_bounds
        # learnable affine parameters
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))
        self.activation = nn.Tanh()

    def _clip(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor, -self.clip_bounds, self.clip_bounds)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Compute attention and return processed output."""
        rotation_params = self._clip(torch.as_tensor(rotation_params, dtype=torch.float32))
        entangle_params = self._clip(torch.as_tensor(entangle_params, dtype=torch.float32))
        inputs = torch.as_tensor(inputs, dtype=torch.float32)

        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key   = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)

        attn_out = scores @ inputs
        attn_out = self.activation(attn_out)
        attn_out = attn_out * self.scale + self.shift
        return attn_out.numpy()
