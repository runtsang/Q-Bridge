import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """Adds a residual connection around two linear layers."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin1 = nn.Linear(in_features, out_features)
        self.lin2 = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        return out + residual

class HybridHead(nn.Module):
    """Classical head that maps features to a 2‑dimensional probability vector."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return F.softmax(logits, dim=-1)

class HybridClassifier(nn.Module):
    """CNN followed by a residual‑enhanced fully‑connected block and a hybrid head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.residual = ResidualBlock(120, 120)
        self.fc3 = nn.Linear(84, 2)
        self.head = HybridHead(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.residual(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.head(x)
        return probs
