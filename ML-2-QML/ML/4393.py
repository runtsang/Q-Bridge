"""Hybrid binary classifier that fuses classical convolution, kernel, and fraud‑detection inspired layers.

The architecture extends the original hybrid QCNet by adding:
* a classical 2×2 convolution filter that acts as a local feature extractor
* an RBF kernel layer that compares the high‑level feature map to a learnable reference vector
* a fraud‑detection style sequential module that emulates the photonic circuit with linear + tanh + scaling

The forward pass returns a two‑class probability distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalConvFilter(nn.Module):
    """2×2 convolution filter implemented with a single learnable weight and bias."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.weight = nn.Parameter(torch.randn(kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch = x[..., : self.kernel_size, : self.kernel_size]
        patch = patch.mean(dim=1, keepdim=True)
        logits = F.conv2d(patch, self.weight.unsqueeze(0).unsqueeze(0), bias=self.bias)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().unsqueeze(-1)


class RBFKernel(nn.Module):
    """Classical RBF kernel that compares a feature vector to a learnable reference."""

    def __init__(self, gamma: float = 1.0, dim: int = 84) -> None:
        super().__init__()
        self.gamma = gamma
        self.reference = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x - self.reference
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class FraudLayer(nn.Module):
    """Linear‑tanh‑scaling layer inspired by the photonic fraud‑detection circuit."""

    def __init__(self, in_features: int, out_features: int, clip: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.activation = nn.Tanh()
        self.clip = clip
        if self.clip:
            self.linear.weight.data.clamp_(-5.0, 5.0)
            self.linear.bias.data.clamp_(-5.0, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        scale = torch.tensor([1.0, 1.0], device=x.device)
        shift = torch.tensor([0.0, 0.0], device=x.device)
        return out * scale + shift


class FraudDetectionModel(nn.Sequential):
    """Sequential fraud‑detection style network."""

    def __init__(self, layers: int = 3, clip: bool = True) -> None:
        modules = [FraudLayer(2, 2, clip=False)]
        modules += [FraudLayer(2, 2, clip=clip) for _ in range(layers - 1)]
        modules.append(nn.Linear(2, 1))
        super().__init__(*modules)


class HybridBinaryClassifier(nn.Module):
    """Classical hybrid binary classifier that integrates convolution, kernel, and fraud‑detection modules."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)

        # Auxiliary modules
        self.conv_filter = ClassicalConvFilter()
        self.kernel = RBFKernel(gamma=1.0, dim=84)
        self.fraud_model = FraudDetectionModel(layers=3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Backbone
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        fc2_out = F.relu(self.fc2(x))  # shape (batch, 84)

        # Conv filter on raw input
        conv_out = self.conv_filter(inputs)  # shape (batch, 1)

        # Kernel similarity
        kernel_out = self.kernel(fc2_out)  # shape (batch, 1)

        # Combine auxiliary features
        aux = torch.cat((conv_out, kernel_out), dim=-1)  # shape (batch, 2)

        # Fraud‑detection style processing
        fraud_logits = self.fraud_model(aux)  # shape (batch, 1)

        probs = torch.sigmoid(fraud_logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridBinaryClassifier"]
