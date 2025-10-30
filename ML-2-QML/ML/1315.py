import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionBlock(nn.Module):
    """Simple attention that learns a weight per feature."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Element‑wise attention weighting
        return x * self.weight

class CalibrationModule(nn.Module):
    """Adds a learnable bias to the output."""
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias

class HybridQuantumBinaryClassifier(nn.Module):
    """Classical CNN with attention and calibration, mirroring the quantum hybrid."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.attention = AttentionBlock(84)
        self.calibration = CalibrationModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.attention(x)
        x = self.fc3(x)
        x = self.calibration(x)
        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
