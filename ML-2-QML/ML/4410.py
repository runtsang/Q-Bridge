from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class QuantumHybridClassifier(nn.Module):
    """
    Hybrid classical classifier that mirrors a quantum convolutional neural network.
    It consists of a 2‑D convolutional feature extractor (like QFCModel),
    a stack of fully‑connected layers (like QCNNModel), and a final softmax sampler
    (like SamplerQNN) producing class probabilities.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, num_classes),
        )
        # Sampler (softmax)
        self.sampler = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features → flatten → fully connected → sampler.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sampler(x)

__all__ = ["QuantumHybridClassifier"]
