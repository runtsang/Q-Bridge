from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridBinaryClassifier(nn.Module):
    """
    Classical binary classifier with optional quantum head.

    In the classical mode the network ends with a linear layer followed by a sigmoid.
    The quantum head is not defined here; use the qml module for hybrid experiments.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor – same topology as QCNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        # Fully‑connected projection
        self.fc = nn.Sequential(
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )
        # Classical head
        self.classic_head = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        logits = self.classic_head(x)
        # return probability vector [p, 1-p]
        return torch.cat([logits, 1 - logits], dim=-1)

__all__ = ["HybridBinaryClassifier"]
