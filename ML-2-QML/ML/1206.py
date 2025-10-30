import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResidualBlock(nn.Module):
    """Adds a residual connection around a 1x1 convolution."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.bn(self.conv(x)))

class SqueezeExpand(nn.Module):
    """Squeeze‑and‑expand block that reduces then restores channel dimension."""
    def __init__(self, in_ch: int, squeeze_factor: int = 4) -> None:
        super().__init__()
        squeezed = in_ch // squeeze_factor
        self.squeeze = nn.Conv2d(in_ch, squeezed, kernel_size=1)
        self.expand = nn.Conv2d(squeezed, in_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(F.relu(self.squeeze(x)))

class Hybrid(nn.Module):
    """Classical dense head that replaces the quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)

class QCNet(nn.Module):
    """CNN‑based binary classifier with residual and squeeze‑expand blocks."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.res1 = ResidualBlock(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.res2 = ResidualBlock(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.sqexp = SqueezeExpand(15)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(1, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = self.pool(x)
        x = self.drop1(x)
        x = self.sqexp(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["ResidualBlock", "SqueezeExpand", "Hybrid", "QCNet"]
