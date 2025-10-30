from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuantumBinaryClassifier(nn.Module):
    """Classical counterpart of the hybrid quantum binary classifier.

    Combines a convolutional backbone, a linear feature extractor, and a
    SamplerQNN‑style head that outputs a two‑class probability distribution.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone mimicking the original QCNet
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
            nn.Flatten()
        )
        # Feature MLP
        self.fc = nn.Sequential(
            nn.Linear(55815, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True)
        )
        # SamplerQNN‑style head
        self.head = nn.Sequential(
            nn.Linear(84, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc(x)
        logits = self.head(x)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["HybridQuantumBinaryClassifier"]
