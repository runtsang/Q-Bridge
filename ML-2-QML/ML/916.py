import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumHybridClassifier(nn.Module):
    """Classical CNN backbone with a sigmoid head, serving as the classical counterpart
    to the hybrid quantum classifier.  The architecture mirrors the seed model
    but replaces the quantum expectation layer with a dense sigmoid head.
    Additionally, a focal loss helper is provided for confidence‑aware training.
    """
    def __init__(self):
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()

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
        probs = self.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

    @staticmethod
    def focal_loss(preds: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
        """Compute focal loss for binary classification."""
        probs = preds[:, 0]
        targets = targets.float()
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = -((1 - pt) ** gamma) * torch.log(pt + eps)
        return loss.mean()

__all__ = ["QuantumHybridClassifier"]
