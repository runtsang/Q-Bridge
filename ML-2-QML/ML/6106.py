"""
Hybrid self‑attention + EstimatorQNN implementation for classical inference.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

class HybridSelfAttentionEstimator:
    """
    Combines a classical self‑attention block with a small feed‑forward regressor.
    """

    def __init__(self, embed_dim: int = 4, hidden_dim: int = 8) -> None:
        self.embed_dim = embed_dim
        self.attention = self._build_attention()
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _build_attention(self) -> nn.Module:
        """
        Builds a lightweight self‑attention module using torch tensors.
        """
        class ClassicalSelfAttention(nn.Module):
            def __init__(self, embed_dim: int) -> None:
                super().__init__()
                self.embed_dim = embed_dim
                # Random rotation and entangle matrices for demonstration
                self.rotation = nn.Parameter(
                    torch.randn(embed_dim, embed_dim, dtype=torch.float32)
                )
                self.entangle = nn.Parameter(
                    torch.randn(embed_dim, embed_dim, dtype=torch.float32)
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                # inputs: (batch, embed_dim)
                query = inputs @ self.rotation
                key = inputs @ self.entangle
                scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
                return scores @ inputs

        return ClassicalSelfAttention(self.embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run the hybrid pipeline: attention → regression.
        """
        attn_out = self.attention(inputs)
        return self.regressor(attn_out)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper for numpy input.
        """
        with torch.no_grad():
            tensor = torch.as_tensor(inputs, dtype=torch.float32)
            output = self.forward(tensor)
        return output.squeeze().numpy()

__all__ = ["HybridSelfAttentionEstimator"]
