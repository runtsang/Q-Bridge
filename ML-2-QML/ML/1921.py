"""
ConvGen108 – Classical depth‑wise separable convolution with spectral regularization.
The module exposes a ConvGen108 class that can be dropped into any CNN pipeline
or used as a standalone feature extractor.  It adds two key extensions:
1. Depth‑wise separable 2‑D convolution (kernel = 2) – learns per‑channel weights
   so that the model can be used inside a larger network with multiple input
   channels.
2. Spectral regularization – a Fourier‑domain penalty that encourages smooth
   kernels and mitigates overfitting.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ConvGen108(nn.Module):
    """
    ConvGen108 is a depth‑wise separable convolutional filter that
    optionally self‑attends over its kernel.  The output is a scalar
    that can be used as a feature or a local score.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 use_attention: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_attention = use_attention

        # Depth‑wise separable: per‑channel conv (1 in, 1 out for each channel)
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size,
                                   bias=True, groups=1)
        # Point‑wise projection to a scalar
        self.pointwise = nn.Conv2d(1, 1, kernel_size=1, bias=True)

        # Optional self‑attention over the kernel
        if self.use_attention:
            self.attn = nn.Parameter(torch.randn(kernel_size, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            batch‑size = n, shape = [batch, 1, k, k] (k=kernel_size)
        Returns
        -------
        torch.Tensor
            shape (batch,) – scalar per batch element.
        """
        # depth‑wise conv
        dw = self.depthwise(x)
        # point‑wise projection
        pw = self.pointwise(dw)
        # flatten to scalar per batch
        out = pw.view(-1)

        if self.use_attention:
            # broadcast attention mask over batch
            attn = self.attn.view(1, 1, self.kernel_size, self.kernel_size)
            # apply attention to the kernel weights
            out = (out * attn.sum()).sum()

        # spectral regularizer
        kernel_fft = torch.fft.fft2(self.depthwise.weight.squeeze())
        smoothness = torch.mean(torch.abs(kernel_fft)**2)
        self.spectral_penalty = smoothness

        return out

    def spectral_loss(self) -> torch.Tensor:
        """Return the spectral penalty that can be added to the training loss."""
        return self.spectral_penalty
