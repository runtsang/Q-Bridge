"""Purely classical binary classifier with CNN backbone and fully‑connected head.

The architecture is inspired by the QCNN and Quantum‑NAT examples, providing a strong baseline for comparison with the hybrid quantum version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridBinaryClassifier(nn.Module):
    """
    Classical classifier with a CNN backbone and fully‑connected head.

    Architecture:
        - 3 conv layers (3→8→16→32 channels)
        - BatchNorm2d + ReLU after each conv
        - MaxPool2d(2) after each conv
        - Linear head: 32×4×4 → 128 → 64 → 2
        - Dropout for regularisation
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.head(x)
        return F.softmax(logits, dim=-1)

__all__ = ["HybridBinaryClassifier"]
