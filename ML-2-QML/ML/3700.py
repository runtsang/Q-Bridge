"""ConvHybridNet: classical implementation that mimics the quantum filter and
   provides a drop‑in replacement for quanvolution layers."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Classical 2‑D convolutional filter that emulates the quantum filter.

    Parameters
    ----------
    kernel_size: int
        Size of the convolution kernel.
    threshold: float
        Threshold for the sigmoid activation.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Compute the mean activation of the filter.

        Parameters
        ----------
        data: torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Mean activation scalar for each sample in the batch.
        """
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])

class HybridHead(nn.Module):
    """Hybrid head that replaces the quantum expectation layer with a
    differentiable sigmoid activation.
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return torch.sigmoid(logits + self.shift)

class ConvHybridNet(nn.Module):
    """Convolutional network followed by a hybrid head.

    The architecture mirrors the quantum version but replaces the quantum
    circuit with a classical dense head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = HybridHead(in_features=1, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["ConvFilter", "HybridHead", "ConvHybridNet"]
