"""
Classical neural network for binary classification.
It extends the original seed by adding a residual dense block and a
calibrated sigmoid head, and provides train() and save() helpers for
quick experimentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

__all__ = ["HybridFunction", "Hybrid", "QCNet", "ResidualBlock"]

class ResidualBlock(nn.Module):
    """Two‑layer dense residual block with ReLU activations."""
    def __init__(self, features: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(features, features)
        self.lin2 = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.lin2(F.relu(self.lin1(x))))

class HybridFunction(nn.Module):
    """Simple sigmoid head used instead of a quantum circuit."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)

class Hybrid(nn.Module):
    """Linear head that mirrors the shape of the original quantum layer."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze(-1)
        return torch.sigmoid(logits + self.shift)

class QCNet(nn.Module):
    """CNN → residual dense block → hybrid head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.residual = ResidualBlock(84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(1, shift=0.0)

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
        x = self.residual(x)
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

    def train_one_epoch(self, dataloader: Iterable, optimizer, criterion, device: torch.device):
        self.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
