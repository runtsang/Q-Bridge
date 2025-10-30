import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """ResNet‑like residual block for efficient feature reuse."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)

class QuantumHybridNAT(nn.Module):
    """Hybrid CNN–quantum model for binary classification (classical implementation)."""
    def __init__(self, n_qubits: int = 4, shift: float = 0.0):
        super().__init__()
        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(128, 128, stride=1),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        # Classical head that mimics a quantum expectation
        self.head = nn.Linear(128, 1, bias=False)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        logits = self.head(features)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["QuantumHybridNAT"]
