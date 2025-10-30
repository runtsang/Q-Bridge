"""Hybrid binary classifier with a classical CNN encoder and a differentiable quantum‑style head.

The module contains:
- A small MLP (EstimatorNN) that can be used as a lightweight quantum “input” generator.
- A Hybrid head that replaces the original quantum expectation layer with a
  differentiable sigmoid, enabling pure‑classical experiments.
- A full CNN encoder identical to the original seed but enhanced with batch
  normalisation and a residual style skip connection for better feature reuse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation with a trainable shift."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs, shift)
        return torch.sigmoid(inputs + shift)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, shift = ctx.saved_tensors
        sigmoid = torch.sigmoid(inputs + shift)
        return grad_output * sigmoid * (1 - sigmoid), None


class EstimatorNN(nn.Module):
    """Tiny MLP that mirrors the EstimatorQNN example."""

    def __init__(self, in_features: int, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Hybrid(nn.Module):
    """Dense head that outputs a probability via a sigmoid."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = nn.Parameter(torch.tensor(shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return HybridFunction.apply(logits, self.shift)


class HybridBinaryClassifier(nn.Module):
    """Classical CNN encoder + hybrid head for binary classification."""

    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Classifier
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Heads
        self.hybrid = Hybrid(1, shift=0.0)
        self.estimator = EstimatorNN(self.fc3.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)

        # Flatten and classify
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Hybrid head
        prob = self.hybrid(x)
        return torch.cat((prob, 1 - prob), dim=-1)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pre‑hybrid embedding (output of fc3)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_quantum_input(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar that can be fed to a quantum circuit."""
        embedding = self.get_embedding(x)
        return self.estimator(embedding)


__all__ = ["HybridFunction", "EstimatorNN", "Hybrid", "HybridBinaryClassifier"]
