"""Classical quanvolution filter with RBF kernel expansion.

This module implements a feature extractor that mimics the behaviour of
the original ``QuanvolutionFilter`` but replaces the quantum kernel with a
classical radial‑basis function expansion.  The extracted features are
suitable for a linear head and can be trained end‑to‑end with PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFKernel(nn.Module):
    """Classical RBF kernel expansion.

    Parameters
    ----------
    gamma : float
        Width of the Gaussian kernel.
    n_centers : int
        Number of fixed centers in the feature space.  They are randomly
        initialised and treated as buffers, so they are not trainable.
    """
    def __init__(self, gamma: float = 1.0, n_centers: int = 16):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("centers", torch.randn(n_centers, 4))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF similarity of each patch to all centres.

        Parameters
        ----------
        patches : torch.Tensor
            Shape ``(batch, n_patches, 4)`` – the 4‑dimensional representation
            of each 2×2 pixel patch.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_patches, n_centers)`` – the kernel feature map.
        """
        diff = patches[:, :, None, :] - self.centers[None, None, :, :]
        dist_sq = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * dist_sq)


class QuanvolutionFilter(nn.Module):
    """Classical quanvolution filter with RBF kernel expansion.

    The filter first runs a 2×2 convolution with stride 2 (as in the
    original quanvolution) to obtain a 4‑channel feature map.  Each 2×2
    patch is then mapped into a kernel feature space via :class:`RBFKernel`.
    """
    def __init__(self, gamma: float = 1.0, n_centers: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.kernel = RBFKernel(gamma, n_centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Grayscale images of shape ``(batch, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Flattened kernel feature map of shape ``(batch, 196 * n_centers)``.
        """
        conv_out = self.conv(x)                     # (b, 4, 14, 14)
        patches = conv_out.permute(0, 2, 3, 1)       # (b, 14, 14, 4)
        patches = patches.reshape(x.size(0), -1, 4)  # (b, 196, 4)
        features = self.kernel(patches)             # (b, 196, n_centers)
        return features.reshape(x.size(0), -1)      # (b, 196 * n_centers)


class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network with a classical quanvolution filter.

    The classifier applies the :class:`QuanvolutionFilter` followed by a
    linear head.  The output is log‑softmax, matching the behaviour of
    the original implementation.
    """
    def __init__(self, gamma: float = 1.0, n_centers: int = 16, n_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter(gamma, n_centers)
        self.linear = nn.Linear(n_centers * 196, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
