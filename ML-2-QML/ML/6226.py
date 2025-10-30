"""Classical CNN + fully connected model with multi‑scale feature extraction.

The architecture:
- conv1: 1->8 channels, kernel 3, stride 1, padding 1, ReLU, MaxPool2d(2)
- conv2: 8->16 channels, kernel 3, stride 1, padding 1, ReLU, MaxPool2d(2)
- conv3: 16->32 channels, kernel 3, stride 1, padding 1, ReLU, MaxPool2d(2)
- flatten
- fc1: 32*3*3 -> 128
- dropout 0.5
- fc2: 128 -> 4
- batchnorm on output

The output dimension remains 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QFCModel(nn.Module):
    """Hybrid classical CNN with 3 convolutional stages and a fully connected head."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.norm(x)

__all__ = ["QFCModel"]
