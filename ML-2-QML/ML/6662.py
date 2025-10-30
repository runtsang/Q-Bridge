"""Classical counterpart to the hybrid quantum binary classifier.

This module implements a convolutional backbone followed by a purely classical
dense head that mirrors the quantum expectation head of the quantum module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridBinaryClassifier(nn.Module):
    """Classical binary classifier with a convolutional backbone and a dense head.

    Attributes
    ----------
    backbone : nn.Sequential
        Convolutional feature extractor identical to the quantum version.
    classifier : nn.Linear
        Dense head mapping the backbone features to two logits.
    """

    def __init__(self, in_channels: int = 3, num_features: int = 120) -> None:
        super().__init__()
        # Convolutional backbone identical to the quantum implementation
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_features),
            nn.ReLU(),
        )
        # Dense head
        self.classifier = nn.Linear(num_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["HybridBinaryClassifier"]
