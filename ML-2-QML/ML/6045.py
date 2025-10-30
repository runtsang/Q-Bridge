"""Classical depth‑wise separable convolutional network with residual connection.

The network replaces the fixed 2×2 convolution of the original QuanvolutionFilter with a
depth‑wise separable convolution followed by a point‑wise convolution, a batch‑norm and
ReLU, and adds a residual shortcut from the input to the conv block.  A global
average pooling is omitted; the flattened feature map is directly fed into a linear
classifier.  This design keeps the overall input/output shape identical to the
original but provides more expressive power and a residual pathway that eases
gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """Depth‑wise separable convolution + residual branch for MNIST."""

    def __init__(self) -> None:
        super().__init__()
        # Depth‑wise separable convolution
        self.depthwise = nn.Conv2d(1, 1, kernel_size=2, stride=2, groups=1, bias=False)
        self.pointwise = nn.Conv2d(1, 4, kernel_size=1, bias=False)
        # Residual projection to match channel dimension (1->4) and spatial size (28->14)
        self.residual = nn.Conv2d(1, 4, kernel_size=1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        # Linear classifier
        self.classifier = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual path
        res = self.residual(x)
        # Conv path
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        # Add residual
        out = out + res
        out = self.relu(out)
        # Flatten
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
