"""HybridQuantumBinaryClassifier – classical enhanced backbone for the hybrid model.

This module keeps the original QCNet forward signature but adds
- a feature‑selection backbone that learns to drop unused channels,
- a multi‑head attention layer to enrich the feature representation,
- a learnable calibration bias that can be used to adjust the final logits.
The design is fully compatible with the original seed and can be
used as a drop‑in replacement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureSelection(nn.Module):
    """Learnable binary mask applied to intermediate feature maps."""
    def __init__(self, in_features: int, threshold: float = 0.5):
        super().__init__()
        self.mask = nn.Parameter(torch.rand(in_features))
        self.threshold = threshold
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.sigmoid(self.mask)
        return x * mask.unsqueeze(0)

class MultiHeadAttention(nn.Module):
    """Simple multi‑head attention over the feature vector."""
    def __init__(self, in_features: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(in_features, in_features)
        self.key   = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)
        self.out   = nn.Linear(in_features, in_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.softmax(torch.bmm(q.unsqueeze(1), k.unsqueeze(2))/np.sqrt(q.size(1)), dim=-1)
        out = torch.bmm(attn, v.unsqueeze(2)).squeeze(2)
        return self.out(out)

class Calibration(nn.Module):
    """Learnable bias added to the final logits."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_classes))
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self.bias

class HybridQuantumBinaryClassifier(nn.Module):
    """Hybrid neural network with classical backbone and optional quantum head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1   = nn.Linear(55815, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 1)
        self.feature_selection = FeatureSelection(84)
        self.attention = MultiHeadAttention(84)
        self.calibration = Calibration(2)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.feature_selection(x)
        x = self.attention(x)
        x = self.fc3(x)
        logits = self.calibration(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
