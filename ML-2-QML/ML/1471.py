import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualSeparableBlock(nn.Module):
    """Depth‑wise separable residual block."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch!= out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pw(out)
        out = self.bn2(out)
        out = self.relu(out)
        return self.relu(out + self.res(x))

class QFCHybridModel(nn.Module):
    """Hybrid classical CNN with residual depth‑wise separable convs."""
    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualSeparableBlock(in_channels, 16),
            ResidualSeparableBlock(16, 32),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        out = self.fc(features)
        return self.norm(out)

__all__ = ["QFCHybridModel"]
