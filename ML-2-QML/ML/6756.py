"""Hybrid classical convolution and self‑attention module.

This module merges the ConvFilter and ClassicalSelfAttention classes
from the original seed files into a single pipeline. The convolution
stage extracts local features, while the attention stage aggregates
them globally. The design allows easy switching between classical
and quantum backends by simply instantiating the appropriate class.
"""

from __future__ import annotations

import torch
import numpy as np
from torch import nn

def HybridConvAttention(embed_dim: int = 4,
                        kernel_size: int = 2,
                        threshold: float = 0.0) -> nn.Module:
    """
    Return a PyTorch module that applies a 2‑D convolution followed
    by a classical self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the attention space.
    kernel_size : int
        Size of the convolution kernel.
    threshold : float
        Threshold used in the sigmoid activation after convolution.
    """
    class ConvAttention(nn.Module):
        def __init__(self):
            super().__init__()
            # Convolution stage
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
            self.threshold = threshold
            # Attention parameters
            self.embed_dim = embed_dim
            self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
            self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through convolution and self‑attention.

            Parameters
            ----------
            data : torch.Tensor
                Input image of shape (H, W) or (B, 1, H, W).

            Returns
            -------
            torch.Tensor
                Output after attention, same spatial shape as input.
            """
            if data.dim() == 2:
                data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            logits = self.conv(data)
            act = torch.sigmoid(logits - self.threshold)
            act_flat = act.view(act.size(0), -1)  # flatten spatial dims

            # Classical self‑attention
            query = act_flat @ self.rotation_params
            key   = act_flat @ self.entangle_params
            scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
            attended = scores @ act_flat
            return attended.view(act.size(0), 1, *act.size()[2:])  # reshape back

        def run(self, data: np.ndarray) -> np.ndarray:
            """
            Run the module on a NumPy array.

            Parameters
            ----------
            data : np.ndarray
                2‑D array of shape (H, W) or (B, H, W).

            Returns
            -------
            np.ndarray
                Processed array, same shape as input.
            """
            tensor = torch.as_tensor(data, dtype=torch.float32)
            out = self.forward(tensor).detach().cpu().numpy()
            return out.squeeze()

    return ConvAttention()
__all__ = ["HybridConvAttention"]
