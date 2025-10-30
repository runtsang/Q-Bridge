"""
Depth‑wise separable quanvolution filter with a classical post‑processing head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """
    Depth‑wise separable quanvolution filter followed by a linear classifier.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 kernel_size: int = 2,
                 stride: int = 2,
                 bias: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        # Depth‑wise convolution: one filter per input channel
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels * out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   groups=in_channels,
                                   bias=bias)
        # Point‑wise convolution to reduce channel dimension
        self.pointwise = nn.Conv2d(in_channels * out_channels,
                                   out_channels,
                                   kernel_size=1,
                                   bias=bias)
        # Optional dropout for regularisation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Learnable bias per patch (4 per 2x2 patch)
        self.patch_bias = nn.Parameter(torch.zeros(out_channels))
        # Classifier head
        # 14x14 patches, each with 4 channels
        self.linear = nn.Linear(out_channels * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, 10).
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = out + self.patch_bias.view(1, -1, 1, 1)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
