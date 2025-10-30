from __future__ import annotations

import torch
import numpy as np
from torch import nn

class HybridConvAttention(nn.Module):
    """
    Classical hybrid layer that first applies a 2‑D convolution and then a
    self‑attention mechanism on the resulting feature map.  Designed as a
    drop‑in replacement for the original Conv and SelfAttention modules
    while providing richer representational power.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel (number of pixels).
    conv_threshold : float, default 0.0
        Threshold used for the sigmoid activation after convolution.
    embed_dim : int, default 4
        Dimensionality of the self‑attention embedding.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        embed_dim: int = 4,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.embed_dim = embed_dim

        # Convolutional filter
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Linear projections for self‑attention
        self.query_proj = nn.Linear(1, embed_dim)
        self.key_proj   = nn.Linear(1, embed_dim)

    def forward(self, input_patch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input_patch : torch.Tensor
            Tensor of shape (H, W) representing a single kernel‑sized patch.

        Returns
        -------
        torch.Tensor
            The attended feature, shape (embed_dim,).
        """
        # Convolution step
        patch = input_patch.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        logits = self.conv(patch)
        activations = torch.sigmoid(logits - self.conv_threshold)
        flat = activations.view(-1)  # (kernel_size*kernel_size,)

        # Self‑attention step
        query = self.query_proj(flat.unsqueeze(0))   # (1,embed_dim)
        key   = self.key_proj(flat.unsqueeze(0))     # (1,embed_dim)
        value = flat.unsqueeze(0)                    # (1,features)

        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        attended = scores @ value
        return attended.squeeze(0)

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper to accept NumPy arrays.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        np.ndarray
            1‑D array of shape (embed_dim,).
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        result = self.forward(tensor)
        return result.numpy()

__all__ = ["HybridConvAttention"]
