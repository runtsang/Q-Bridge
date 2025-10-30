import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridFunction(nn.Module):
    """Differentiable sigmoid head with optional shift."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits + self.shift)

class HybridQuantumBinaryClassifier(nn.Module):
    """
    Classical CNN backbone followed by a hybrid head.
    In the ML variant, the head is a simple linear layer with a sigmoid activation.
    """
    def __init__(self, shift: float = 0.0):
        super().__init__()
        # CNN backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Head
        self.head = HybridFunction(shift=shift)

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
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
