"""Enhanced classical quanvolution pipeline.

The new class `QuanvolutionGen139` implements a two‑stage
pipeline: first, a small convolutional block that extracts
local patches, then a trainable MLP that maps the feature
vector to the logits.  The MLP can be applied
to the quantum‑style patches or to the raw image
directly.  It also includes optional batch‑norm
and dropout for regularisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionGen139(nn.Module):
    """Feature‑extractor + MLP head with optional batch‑norm."""
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 conv_kernel: int = 2,
                 conv_stride: int = 2,
                 mlp_hidden: int = 128,
                 num_classes: int = 10,
                 use_batchnorm: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=conv_kernel, stride=conv_stride)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        # After conv, image size becomes 14x14, features: out_channels*14*14
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 14 * 14, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        logits = self.mlp(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen139"]
