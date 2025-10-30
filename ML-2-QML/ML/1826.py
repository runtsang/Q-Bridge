"""
Extended quanvolution filter with a learnable classical residual block and a linear head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """
    Classical implementation of the Quanvolution hybrid network.

    The architecture consists of:
      * A 2×2 convolution that extracts non‑overlapping patches from the input image.
      * A residual block that processes each patch while preserving dimensionality.
      * A linear classifier that maps the concatenated patch features to class logits.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        # 2×2 convolution to extract 4 feature maps per patch
        self.patch_conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2, bias=False)

        # Residual block: 3×3 convs with batch norm and ReLU
        self.res_block = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
        )

        # Linear classifier
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax probabilities of shape (batch_size, num_classes).
        """
        # Extract 2×2 patches via strided convolution
        patches = self.patch_conv(x)  # shape: (B, 4, 14, 14)

        # Apply residual block with skip connection
        residual = patches
        out = self.res_block(patches)
        out = out + residual
        out = F.relu(out)

        # Flatten and classify
        features = out.view(x.size(0), -1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
