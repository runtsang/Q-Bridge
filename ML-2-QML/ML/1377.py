"""
HybridClassifier – classical baseline with attention and a linear head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAttention(nn.Module):
    """Learnable attention over flattened features."""
    def __init__(self, in_features: int, hidden_size: int = 32):
        super().__init__()
        self.attn = nn.Linear(in_features, hidden_size)
        self.proj = nn.Linear(hidden_size, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = F.relu(self.attn(x))
        weights = F.softmax(self.proj(attn), dim=1)
        return x * weights

class HybridClassifier(nn.Module):
    """Classical CNN with attention and a sigmoid head."""
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
        self.attention = FeatureAttention(55815)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.attention(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        probs = self.sigmoid(x)
        return torch.cat((probs.unsqueeze(-1), 1 - probs.unsqueeze(-1)), dim=-1)

__all__ = ["FeatureAttention", "HybridClassifier"]
