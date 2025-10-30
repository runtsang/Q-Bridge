import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChannelAttention(nn.Module):
    """Channel‑wise attention module used before flattening."""
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Hybrid(nn.Module):
    """Simple dense head that replaces the quantum circuit."""
    def __init__(self, in_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return torch.sigmoid(self.linear(x))

class QCNet(nn.Module):
    """Enhanced classical CNN with attention pooling and a lightweight hybrid head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.attention = ChannelAttention(15)
        # compute flatten size with a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            x = self.conv1(dummy)
            x = self.pool(x)
            x = self.drop1(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.drop1(x)
            x = self.attention(x)
            x = torch.flatten(x, 1)
            self.flatten_size = x.shape[1]
        self.fc1 = nn.Linear(self.flatten_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.attention(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        probs = self.hybrid(x)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["ChannelAttention", "Hybrid", "QCNet"]
