from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ConvKernelFilter(nn.Module):
    """Hybrid convolutional filter that augments classical conv output with an RBF kernel similarity.

    The module first applies a 2‑D convolution followed by a sigmoid and mean reduction.  The resulting scalar
    is then modulated by an RBF kernel similarity between the flattened convolution result and a
    reference vector.  The reference vector can be learned or fixed; here it is a trainable parameter.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, gamma: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.gamma = gamma
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        # Reference vector used in the kernel; shape matches flattened conv output
        self.register_parameter("ref_vec", nn.Parameter(torch.zeros(kernel_size * kernel_size)))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            2‑D input of shape (kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Scalar output combining conv activation and kernel similarity.
        """
        # Ensure correct shape
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        conv_out = activations.mean()

        # Flatten conv output to a vector for kernel computation
        flat = activations.view(-1)
        diff = flat - self.ref_vec
        kernel_sim = torch.exp(-self.gamma * torch.sum(diff * diff))

        return conv_out * kernel_sim

def Conv() -> ConvKernelFilter:
    """Return a drop‑in replacement for the original Conv filter that also incorporates
    an RBF kernel similarity.  The function signature matches the original ``Conv`` so
    existing code can import it unchanged."""
    return ConvKernelFilter()

__all__ = ["Conv"]
