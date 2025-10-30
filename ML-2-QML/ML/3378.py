import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RBFKernelLayer(nn.Module):
    """Compute an RBF kernel between input scalars and learnable support vectors."""
    def __init__(self, in_features: int, n_support: int = 10, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.support = nn.Parameter(torch.randn(n_support, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        diff = x.unsqueeze(1) - self.support.unsqueeze(0)  # (batch, n_support, features)
        dist_sq = (diff ** 2).sum(dim=2)  # (batch, n_support)
        return torch.exp(-self.gamma * dist_sq)

class HybridBinaryClassifier(nn.Module):
    """CNN backbone + learnable RBF kernel + linear head for binary classification."""
    def __init__(self,
                 n_support: int = 10,
                 gamma: float = 1.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.kernel = RBFKernelLayer(in_features=1, n_support=n_support, gamma=gamma)
        self.linear = nn.Linear(n_support, 1)

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
        x = self.fc3(x)
        # kernel layer expects scalar features; collapse to shape (batch, 1)
        x = x.view(x.size(0), 1)
        k = self.kernel(x)  # (batch, n_support)
        logits = self.linear(k)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier", "RBFKernelLayer"]
