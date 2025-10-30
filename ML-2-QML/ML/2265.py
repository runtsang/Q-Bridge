"""Hybrid classical QCNN with quanvolution filtering."""

import torch
from torch import nn
import torch.nn.functional as F

class ClassicalQuanvolutionFilter(nn.Module):
    """2×2 patch convolution followed by flattening."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class ClassicalQCNN(nn.Module):
    """Fully connected network mimicking quantum convolution layers."""
    def __init__(self):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridQCNNClassifier(nn.Module):
    """Combines a quanvolution filter with a classical QCNN head."""
    def __init__(self):
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        self.cnn = ClassicalQCNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        return self.cnn(features)

__all__ = ["HybridQCNNClassifier", "ClassicalQuanvolutionFilter", "ClassicalQCNN"]
