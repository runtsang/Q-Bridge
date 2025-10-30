import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalHybridHead(nn.Module):
    """Classical approximation of a quantum expectation head."""
    def __init__(self, in_features: int, shift: float = np.pi / 2):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        # emulate quantum expectation with a shifted sigmoid
        return torch.sigmoid(logits + self.shift)

class QuantumHybridClassifier(nn.Module):
    """CNN backbone followed by a classical head that mimics quantum behaviour."""
    def __init__(self, num_classes: int = 2, feature_dim: int = 8, shift: float = np.pi / 2):
        super().__init__()
        # feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, feature_dim)
        self.head = ClassicalHybridHead(feature_dim, shift=shift)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        prob = self.head(x)
        return torch.cat([prob, 1 - prob], dim=-1)

__all__ = ["QuantumHybridClassifier"]
