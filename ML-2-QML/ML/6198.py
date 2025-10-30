"""HybridQuantumBinaryClassifier: Classical PyTorch implementation.

This module implements a purely classical binary classifier that mirrors
the architecture of the original hybrid model.  The CNN backbone is
followed by a dense head that produces logits, which are passed through
a sigmoid to obtain class probabilities.  The module can be trained on
a CPU or GPU using standard PyTorch optimisers.
"""

import torch
import torch.nn as nn

class HybridQuantumBinaryClassifier(nn.Module):
    """Purely classical binary classifier with a CNN backbone."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(15, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = self.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
