import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv2d, MaxPool2d, ReLU, Linear

class ResidualBlock(nn.Module):
    """Lightweight residual block used in the deeper CNN."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = BatchNorm2d(channels)
        self.conv2 = Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(channels)
        self.relu = ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class QFCModel(nn.Module):
    """Deeper CNN with residual blocks followed by a flexible output head."""
    def __init__(self, output_dim: int = 4, use_regression: bool = False):
        super().__init__()
        self.features = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2),
            ResidualBlock(32),
            MaxPool2d(2),
            ResidualBlock(64),
            MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            Linear(64, 128),
            ReLU(),
            nn.Dropout(0.5),
            Linear(128, output_dim)
        )
        self.norm = nn.BatchNorm1d(output_dim)
        self.use_regression = use_regression
        self.loss_fn = nn.MSELoss() if use_regression else nn.CrossEntropyLoss()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.global_pool(features).view(x.shape[0], -1)
        out = self.fc(pooled)
        out = self.norm(out)
        return out

__all__ = ["QFCModel"]
