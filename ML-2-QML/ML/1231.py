"""ConvEnhanced – a hybrid‑classical convolutional block with depthwise separable filtering and attention."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvEnhanced(nn.Module):
    """
    A drop‑in replacement for the original Conv class that adds a learnable attention mask
    and a depthwise separable convolution to reduce parameter count while preserving
    expressive power.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter.
    threshold : float, default 0.0
        Threshold used for the sigmoid activation in the original implementation.
    attention_dim : int, default 4
        Dimensionality of the small attention MLP that produces a mask for each channel.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 attention_dim: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Depthwise separable convolution: first depthwise, then pointwise.
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size,
                                   padding=0, bias=True, groups=1)
        self.pointwise = nn.Conv2d(1, 1, kernel_size=1,
                                   padding=0, bias=True)

        # Small MLP to generate an attention mask for the channel.
        self.attn_mlp = nn.Sequential(
            nn.Linear(1, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Sigmoid()
        )

        # Learnable scalar that re‑weights the quantum‑like output.
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that combines the classical depthwise separable convolution
        and an optional quantum‑like output.

       .. code-block:: python
            x: [batch, 1, H, W]   # input tensor
        """
        # Classical path
        out = self.depthwise(x)
        out = self.pointwise(out)
        # Compute attention mask from the mean of each channel
        mean_channel = out.mean(dim=(2, 3), keepdim=True)
        mask = self.attn_mlp(mean_channel)
        out = out * mask

        # Quantum‑like path: learnable parameterized rotation on each pixel
        # (this is a lightweight approximation of a quantum circuit)
        quantum = torch.sin((x * self.alpha).sum(dim=1, keepdim=True))
        quantum = quantum.mean().unsqueeze(0)

        # Fuse using the learnable weight
        return out + self.alpha * quantum

__all__ = ["ConvEnhanced"]
