import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block with optional layer‑norm."""
    def __init__(self, channels: int, use_norm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.LayerNorm([channels, 1, 1]) if use_norm else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        return self.relu(x + out)

class Hybrid(nn.Module):
    """Classical fallback head that maps features to a probability."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = torch.sigmoid(logits + self.shift)
        return probs

class QuantumHybridClassifier(nn.Module):
    """Classic‑only implementation of the hybrid architecture."""
    def __init__(self, num_classes: int = 2, use_residual: bool = True, use_norm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.res1 = ResidualBlock(32, use_norm) if use_residual else nn.Identity()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.res2 = ResidualBlock(64, use_norm) if use_residual else nn.Identity()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.head = Hybrid(128, shift=0.0)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.res2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridClassifier", "ResidualBlock", "Hybrid"]
