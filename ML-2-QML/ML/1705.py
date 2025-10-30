"""Classical PyTorch implementation of a hybrid binary classifier.

The network extends the seed by adding a learnable quantum‑aware dropout
layer and a quantum‑inspired head implemented with a linear projection
followed by a sigmoid activation.  All operations remain fully
classical, but the architecture mirrors the quantum‑aware behaviour
of the QML version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumAwareDropout(nn.Module):
    """Learnable dropout that generates a per‑sample mask from a linear
    projection of the input features."""
    def __init__(self, in_features: int, drop_prob: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_features, 1)
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute a dropout probability for each sample
        prob = torch.sigmoid(self.proj(x))          # (batch, 1)
        mask = torch.bernoulli(1 - prob).to(x.device)  # (batch, 1)
        mask = mask.expand_as(x)                    # broadcast
        # Scale to keep the expectation unchanged
        return x * mask / (1 - self.drop_prob + 1e-7)


class QuantumHybridCNNClassifier(nn.Module):
    """CNN‑based binary classifier that emulates a quantum head."""
    def __init__(self) -> None:
        super().__init__()
        # Convolutional front‑end
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Quantum‑aware dropout
        self.qdrop = QuantumAwareDropout(84, drop_prob=0.1)
        # Quantum‑inspired head
        self.quantum_head = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

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
        x = self.qdrop(x)        # apply learnable dropout
        x = self.fc3(x)          # (batch, 1)
        x = self.quantum_head(x) # (batch, 1)
        return torch.cat((x, 1 - x), dim=-1)


__all__ = ["QuantumHybridCNNClassifier", "QuantumAwareDropout"]
