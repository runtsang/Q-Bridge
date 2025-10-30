"""Classical hybrid binary classifier.

This module implements a convolutional backbone followed by a linear
head that mimics the quantum expectation layer.  The head is a single
trainable linear unit with a sigmoid activation, providing a
classical analogue to the quantum circuit in the QML module.  A small
superposition‑style dataset generator is also exposed for quick
experimentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for binary classification.
    Features are sampled uniformly in [-1, 1] and the label is the sign
    of sin(sum(features)).  The output is a 0/1 label.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    y = (np.sin(x.sum(axis=1)) > 0).astype(np.float32)
    return x, y

class HybridBinaryClassifier(nn.Module):
    """
    Convolutional backbone + linear head for binary classification.
    The head replaces the quantum circuit in the QML implementation
    with a trainable linear layer followed by a sigmoid activation.
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_features: int = 32) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        # Flatten & fully connected layers
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(84, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        # Return probability distribution over two classes
        return torch.cat([x, 1 - x], dim=-1)

__all__ = ["HybridBinaryClassifier", "generate_superposition_data"]
