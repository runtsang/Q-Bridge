import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Single residual block with optional down‑sampling."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class EnhancedQFCModel(nn.Module):
    """Hybrid CNN with residual blocks and a fully‑connected head."""
    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(8, 16, stride=1),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        x = self.features(x)
        x = x.view(bsz, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.norm(x)

__all__ = ["EnhancedQFCModel"]
