import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResDenseBlock(nn.Module):
    """
    Residual dense block: two linear layers with SiLU activation and a skip connection.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm(x)
        return F.silu(x + residual)

class QuantumHybridBinaryClassifier(nn.Module):
    """
    Classical counterpart with residual‑dense post‑processing head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 15 * 15, 120)  # adjust based on input size
        self.fc2 = nn.Linear(120, 84)
        self.res_block = ResDenseBlock(84, 84)
        self.fc3 = nn.Linear(84, 1)

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
        x = self.res_block(x)
        logits = self.fc3(x)
        probs = torch.sigmoid(logits).squeeze(-1)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["QuantumHybridBinaryClassifier"]
