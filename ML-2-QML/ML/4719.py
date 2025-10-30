"""Classical hybrid network combining convolutional quanvolution with a quantum‑inspired head.

The design merges the 2×2 patch convolution from the original Quanvolution,
the simple fully‑connected regression path from EstimatorQNN, and the
quantum‑fully‑connected projection of Quantum‑NAT.  It supports both
classification (default) and regression via the ``regression`` flag.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical two‑pixel patch convolution producing four feature maps."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class QuanvolutionHybrid(nn.Module):
    """Hybrid classical network: quanvolution filter → fully‑connected head."""
    def __init__(self, num_classes: int = 10, regression: bool = False) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        hidden = 128
        self.classifier = nn.Sequential(
            nn.Linear(4 * 14 * 14, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, num_classes if not regression else 1)
        )
        self.regression = regression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)          # [B, 4*14*14]
        logits   = self.classifier(features)
        return logits if self.regression else F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid", "QuanvolutionFilter"]
