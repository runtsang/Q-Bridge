import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout

class ResidualBlock(nn.Module):
    """
    Simple 2‑layer residual block used in the feature extractor.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class QuantumNATGen257(nn.Module):
    """
    A fully‑convolutional neural network with residual blocks and dropout,
    designed to replace the simple CNN of the original Quantum‑NAT.
    The network outputs four features that can be fed into a quantum
    encoder or used directly for classification.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor: two residual conv blocks
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResidualBlock(16, 16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 32),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Projection head
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            Dropout(p=0.2),
            nn.Linear(64, 4),
        )
        self.bn = BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W)

        Returns
        -------
        torch.Tensor
            Normalised feature vector of shape (batch, 4)
        """
        bsz = x.shape[0]
        feat = self.features(x).view(bsz, -1)
        out = self.fc(feat)
        return self.bn(out)

__all__ = ["QuantumNATGen257"]
