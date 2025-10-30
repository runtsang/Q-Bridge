"""Extended CNN with residual blocks and self‑attention for 4‑class output."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple 2‑D residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class SelfAttention(nn.Module):
    """Channel‑wise self‑attention over flattened feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        proj_query = self.query(x).view(b, -1, h * w)
        proj_key = self.key(x).view(b, -1, h * w)
        energy = torch.bmm(proj_query, proj_key.transpose(1, 2))
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(b, -1, h * w)
        out = torch.bmm(proj_value, attention.transpose(1, 2))
        out = out.view(b, c, h, w)
        return self.gamma * out + x


class QFCModelExtended(nn.Module):
    """Residual CNN + self‑attention head producing 4‑dimensional embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.attention = SelfAttention(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.attention(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return self.norm(x)

    @property
    def num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 64‑dimensional penultimate representation."""
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.attention(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return x


__all__ = ["QFCModelExtended"]
