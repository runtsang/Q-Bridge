"""Classical implementation of a hybrid convolutional network for binary classification.

This module mirrors the architecture of the original hybrid model but replaces the
quantum expectation head with a fully‑connected layer or a classical stand‑in
(FCL) to enable fast experimentation and ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCL(nn.Module):
    """Classical stand‑in for a fully‑connected quantum layer.

    The implementation uses a single linear transformation followed by a tanh
    activation, emulating the behaviour of a quantum parameterised circuit that
    would normally output an expectation value.
    """
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))


class HybridQCNet(nn.Module):
    """Convolutional network with an option to attach a classical or hybrid head.

    The default head is a simple linear layer that produces a logit.
    If ``use_hybrid_head`` is True, the head is replaced by an
    FCL instance that simulates quantum behaviour.
    """
    def __init__(self, use_hybrid_head: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        if use_hybrid_head:
            self.head = FCL(self.fc3.out_features)
        else:
            self.head = nn.Linear(self.fc3.out_features, 1)

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
        logits = self.head(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridQCNet", "FCL"]
