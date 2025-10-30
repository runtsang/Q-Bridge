import torch
from torch import nn
import numpy as np
from typing import Sequence

class ConvQuantumKernel(nn.Module):
    """
    Hybrid classical convolution + RBF kernel.

    Combines a differentiable 2‑D convolution (drop‑in for quanvolution) with a
    radial‑basis kernel for pairwise similarity.  The convolution is trained
    end‑to‑end; the kernel is a stateless utility that can be called separately.
    """

    def __init__(self, kernel_size: int = 2, conv_threshold: float = 0.0, gamma: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.gamma = gamma
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolutional filter to a single patch.

        Parameters
        ----------
        patch : torch.Tensor
            Tensor of shape (1, kernel_size, kernel_size) or (kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Normalized activation value.
        """
        if patch.ndim == 2:
            patch = patch.unsqueeze(0).unsqueeze(0)
        logits = self.conv(patch)
        activations = torch.sigmoid(logits - self.conv_threshold)
        return activations.mean()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the RBF Gram matrix between two collections of patches.
        """
        a = [x.reshape(-1) for x in a]
        b = [y.reshape(-1) for y in b]
        a = torch.stack(a)
        b = torch.stack(b)
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        sq_norm = torch.sum(diff * diff, dim=-1)
        kernel = torch.exp(-self.gamma * sq_norm)
        return kernel.numpy()

__all__ = ["ConvQuantumKernel"]
