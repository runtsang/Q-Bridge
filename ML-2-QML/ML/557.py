import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """Classical depth‑wise separable convolutional filter for MNIST images."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2):
        super().__init__()
        # depthwise conv
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   groups=in_channels,
                                   bias=False)
        # pointwise conv
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.fc = nn.Linear(out_channels * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)    # (B, 1, 14, 14)
        x = self.pointwise(x)    # (B, 4, 14, 14)
        x = self.bn(x)
        x = torch.flatten(x, start_dim=1)  # (B, 4*14*14)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
