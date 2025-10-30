import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Lightweight residual block with two Conv2d layers and a skip connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
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

class QuantumNAT(nn.Module):
    """Classical hybrid model inspired by Quantum‑NAT with residual block and temperature‑softmax."""
    def __init__(self, in_channels: int = 1, num_classes: int = 4, temperature: float = 1.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(16, 16),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.avgpool(features).view(x.size(0), -1)
        logits = self.fc(pooled)
        normalized = self.norm(logits)
        probs = F.softmax(normalized / self.temperature, dim=1)
        return probs

__all__ = ["QuantumNAT"]
