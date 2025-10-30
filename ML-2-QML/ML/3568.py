from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid sampler network.

    Combines a learnable 2×2 convolution (with an optional sigmoid
    threshold) and a small two‑layer feed‑forward sampler head.
    The design mirrors the original SamplerQNN architecture while
    adding a convolutional feature extractor inspired by the
    Conv.py seed.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 hidden_dim: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.conv_threshold = conv_threshold
        self.sampler_head = nn.Sequential(
            nn.Linear(kernel_size * kernel_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Softmaxed probability distribution over two classes.
        """
        # Convolution + sigmoid threshold
        conv_out = self.conv(x)
        conv_out = torch.sigmoid(conv_out - self.conv_threshold)
        # Flatten and feed to sampler head
        flat = conv_out.view(conv_out.size(0), -1)
        logits = self.sampler_head(flat)
        return F.softmax(logits, dim=-1)

__all__ = ["HybridSamplerQNN"]
