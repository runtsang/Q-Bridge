"""HybridQuantumClassifier – classical backbone with an extended, trainable head.

The module keeps the original CNN architecture but replaces the
classical “Hybrid” head with a deep multi‑layer perceptron that
provides a multi‑class probability vector.  The new head is
fully differentiable and can be swapped with a quantum module
without changing the user‑side code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveHybrid(nn.Module):
    """An adaptive MLP that replaces the original HybridFunction.

    The head now consists of several fully‑connected layers with
    optional batch‑normalisation and dropout.  A learnable
    bias and a trainable shift scalar are applied before the final
    sigmoid, mirroring the behaviour of the quantum expectation
    layer but giving the model more expressiveness.
    """

    def __init__(self, in_features: int, hidden_dims: list[int] | None = None,
                 dropout: float = 0.0, use_bn: bool = False) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        self.shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x)
        logits = logits + self.shift
        probs = torch.sigmoid(logits)
        return probs


class QCNet(nn.Module):
    """CNN-based binary classifier mirroring the structure of the quantum model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = AdaptiveHybrid(1, hidden_dims=[64, 32], dropout=0.1, use_bn=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
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
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["AdaptiveHybrid", "QCNet"]
