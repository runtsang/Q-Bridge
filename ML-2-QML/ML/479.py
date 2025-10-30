import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumHybridBinaryNet(nn.Module):
    """
    Classical PyTorch implementation of the hybrid quantum binary classifier.
    Replaces the quantum head with a small classical kernel that mimics
    the expectation value of a simple variational circuit.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop = nn.Dropout(0.3)

        # Residual block
        self.res_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.res_bn = nn.BatchNorm2d(64)

        # Fully connected head
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # Classical surrogate for quantum kernel
        self.kernel = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extractor
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)

        # Residual connection
        residual = x
        x = F.relu(self.res_bn(self.res_conv(x)))
        x = x + residual
        x = self.drop(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)  # shape (batch, 1)

        # Classical kernel to produce probability
        prob = self.kernel(x)  # shape (batch, 1)
        # Return two-class probabilities
        return torch.cat([prob, 1 - prob], dim=1)

__all__ = ["QuantumHybridBinaryNet"]
