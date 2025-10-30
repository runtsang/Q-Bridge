"""
HybridClassifier – Classical counterpart with a calibrated sigmoid head.

The module extends the original pure‑classical network by adding a
calibration layer that maps raw logits to probabilities. The
calibration network is a small learnable linear layer followed by a
sigmoid.  A tiny test harness at the bottom demonstrates a forward
pass on random data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibratedHead(nn.Module):
    """
    Trainable calibration head that maps raw logits to a probability
    via a learnable linear transform and a sigmoid.  It mimics the
    behaviour of a quantum expectation head but keeps the model
    fully classical.
    """
    def __init__(self, in_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return probs


class HybridClassifier(nn.Module):
    """
    Classical CNN followed by a small calibration head.  The network
    mirrors the architecture of the original QCNet but replaces the
    quantum expectation layer with a learnable sigmoid head.
    """
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
        self.calibration = CalibratedHead(1)

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
        probs = self.calibration(x)
        return torch.cat((probs, 1 - probs), dim=-1)


# Simple test harness
if __name__ == "__main__":
    model = HybridClassifier()
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    print("Output shape:", output.shape)
    print("Output:", output)


__all__ = ["HybridClassifier", "CalibratedHead"]
