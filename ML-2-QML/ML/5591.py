import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionFilter(nn.Module):
    """Classical 2‑pixel quantum‑inspired filter."""
    def __init__(self, in_channels=1, out_channels=4, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    def forward(self, x):
        return self.conv(x).flatten(1)

class QuantumKernelLayer(nn.Module):
    """RBF‑style kernel as a lightweight classical surrogate for a quantum kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridBinaryClassifier(nn.Module):
    """
    Classical CNN + optional RBF kernel + quanvolution filter.
    Mirrors the interface of the quantum‑ready hybrid model.
    """
    def __init__(self, use_kernel: bool = False, gamma: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.qfilter = QuanvolutionFilter()
        self.fc1 = nn.Linear(6 * 15 * 15 + 4 * 16 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.use_kernel = use_kernel
        if use_kernel:
            self.kernel_layer = QuantumKernelLayer(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # quantum‑inspired feature extraction on raw image
        qfeat = self.qfilter(x)
        # classical CNN path
        x_cnn = F.relu(self.conv1(x))
        x_cnn = self.pool(x_cnn)
        x_cnn = self.dropout(x_cnn)
        x_cnn = torch.flatten(x_cnn, 1)
        # optional kernel interaction
        if self.use_kernel:
            # simple self‑interaction kernel for demonstration
            x_cnn = self.kernel_layer(x_cnn, x_cnn)
        # concatenate features
        x = torch.cat([x_cnn, qfeat], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryClassifier"]
