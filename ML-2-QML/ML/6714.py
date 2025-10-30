import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalResidualBlock(nn.Module):
    """Simple residual block with two conv layers."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out + residual

class ClassicalQuanvolutionFilter(nn.Module):
    """2×2 patch convolution with learnable weights."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        out = self.conv(x)  # (batch, out_channels, 14, 14)
        out = self.relu(self.bn(out))
        return out.view(out.size(0), -1)  # (batch, out_channels*14*14)

class Quanvolution__gen571(nn.Module):
    """Hybrid classical network with residual preprocessing and patch‑based conv."""
    def __init__(self):
        super().__init__()
        self.residual = ClassicalResidualBlock(1)
        self.filter = ClassicalQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual(x)
        features = self.filter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["Quanvolution__gen571"]
