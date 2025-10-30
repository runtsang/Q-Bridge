"""Enhanced classical hybrid network with residual connections and a multi‑head classifier.

This module extends the original seed by adding a residual block after the
convolutional backbone and a lightweight MLP head before the hybrid
activation. The architecture remains compatible with the original
interface: the forward method returns a two‑class probability tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFunction(nn.Module):
    """Sigmoid‑based activation with a learnable bias shift.

    The shift parameter allows the model to adapt the decision boundary
    during training while keeping the operation fully differentiable.
    """
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)

class Hybrid(nn.Module):
    """Dense head that emulates the quantum expectation layer."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.act = HybridFunction(shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return self.act(logits)

class ResidualBlock(nn.Module):
    """Simple residual block with two 3×3 convolutions."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class HybridNet(nn.Module):
    """CNN backbone with a residual block followed by an MLP and a hybrid head."""
    def __init__(self) -> None:
        super().__init__()
        # Backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res   = ResidualBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # MLP head
        self.fc1   = nn.Linear(64 * 4 * 4, 256)
        self.fc2   = nn.Linear(256, 128)

        # Hybrid head
        self.hybrid = Hybrid(128, shift=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.res(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.hybrid(x)
        return torch.cat((logits, 1 - logits), dim=-1)

__all__ = ["HybridFunction", "Hybrid", "HybridNet"]
