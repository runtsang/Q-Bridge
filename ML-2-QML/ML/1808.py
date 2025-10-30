import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionBlock(nn.Module):
    """Simple channel‑wise attention to weight feature maps before the classifier."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).view(x.size(0), -1)
        y = self.fc(y).view(x.size(0), -1, 1, 1)
        return x * y

class HybridFunction(nn.Module):
    """A differentiable wrapper that emulates the quantum expectation with a sigmoid."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(inputs + self.shift)

class QCNet(nn.Module):
    """CNN classifier with a quantum‑inspired hybrid head."""
    def __init__(self):
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.attn = AttentionBlock(15)
        # Classifier
        self.fc1 = nn.Linear(55815, 120)
        self.fc_flat = nn.Flatten()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid_head = HybridFunction(shift=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.attn(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        probs = self.hybrid_head(x)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["AttentionBlock", "HybridFunction", "QCNet"]
