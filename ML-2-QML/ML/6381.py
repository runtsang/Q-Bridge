"""Hybrid classical self‑attention with convolutional feature extraction.

The module mirrors the quantum interface (run(...)) while adding a
learnable convolution before the attention calculation.  The
parameters `rotation_params` and `entangle_params` are treated as
linear maps that produce the query and key matrices, allowing the same
call signature as the quantum implementation.  This design lets the
classical and quantum variants be swapped in a downstream pipeline
without changing the experiment code.
"""
import numpy as np
import torch
from torch import nn

class HybridSelfAttention(nn.Module):
    """Convolution‑augmented self‑attention."""

    def __init__(self, embed_dim: int, kernel_size: int = 3, threshold: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.threshold = threshold

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Linear map for query generation, reshaped to (embed_dim, -1).
        entangle_params : np.ndarray
            Linear map for key generation, reshaped to (embed_dim, -1).
        inputs : np.ndarray
            2‑D array of shape (H, W) or (N, C, H, W).

        Returns
        -------
        np.ndarray
            Attention weighted feature vector.
        """
        tensor = torch.as_tensor(inputs, dtype=torch.float32)
        if tensor.ndim == 2:  # single image
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        conv_out = self.conv(tensor)
        conv_out = torch.sigmoid(conv_out - self.threshold).mean(dim=(2, 3))
        features = conv_out.view(-1, self.embed_dim)

        query = torch.as_tensor(features @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(features @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        value = features
        return (scores @ value).numpy()

def SelfAttention():
    """Return a callable that follows the original API."""
    return HybridSelfAttention(embed_dim=4)

__all__ = ["SelfAttention"]
