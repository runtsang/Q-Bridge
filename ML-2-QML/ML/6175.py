import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depth‑wise separable convolution for efficient feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1,
                                   bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

class ResidualBlock(nn.Module):
    """Simple residual block that adds a depth‑wise convolution to the input."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.relu(self.bn(self.conv(x)))

class QuanvolutionFilter(nn.Module):
    """Classical quanvolution filter using depth‑wise separable convolution and a residual block."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = DepthwiseSeparableConv(1, 4, kernel_size=2, stride=2)
        self.residual = ResidualBlock(4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.residual(y)
        return y.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the quanvolutional filter followed by a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
