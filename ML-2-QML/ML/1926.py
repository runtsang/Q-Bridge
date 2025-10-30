"""HybridQFCModel: a classical convolutional network with residual blocks and advanced training hooks."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A simple residual block with two 3x3 conv layers and dropout."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(F.relu(self.bn2(self.conv2(out))))
        out += self.shortcut(x)
        return F.relu(out)

class HybridQFCModel(nn.Module):
    """HybridQFCModel: a deep CNN with residual blocks, dropout, and a fully connected head.
    The model outputs a 4‑dimensional feature vector, suitable for downstream tasks."""
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32, dropout),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64, dropout),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, 1, H, W)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.norm(out)

__all__ = ["HybridQFCModel"]
