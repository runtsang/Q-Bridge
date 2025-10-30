"""Classical hybrid binary classifier that fuses CNN, quanvolution, and RBF kernel head."""

from __future__ import annotations

import torch
import torch.nn as nn

class QuanvolutionFilter(nn.Module):
    """Classical 2x2 patch-based filter that reduces spatial resolution."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class Kernel(nn.Module):
    """RBF kernel module for classical feature embeddings."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridQuantumBinaryClassifier(nn.Module):
    """Hybrid classical network that uses a CNN backbone, quanvolution filter, and RBF kernel head."""

    def __init__(self, num_support: int = 10, gamma: float = 1.0) -> None:
        super().__init__()
        # CNN backbone adapted to accept 4 input channels from quanvolution
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.quanvolution = QuanvolutionFilter()
        self.flatten = nn.Flatten()
        # Linear projection to reduce dimensionality for kernel
        self.proj = nn.Linear(375, 4)
        # Support vectors for kernel computation
        self.support = nn.Parameter(torch.randn(num_support, 4))
        self.kernel = Kernel(gamma)
        self.linear = nn.Linear(num_support, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convert to grayscale
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.quanvolution(x)  # shape: [batch, 784]
        x = x.view(x.size(0), 4, 14, 14)
        x = self.backbone(x)
        x = self.flatten(x)  # shape: [batch, 375]
        x = self.proj(x)  # shape: [batch, 4]
        # Compute RBF kernel between each sample and support vectors
        batch_size = x.shape[0]
        kernel_features = []
        for i in range(batch_size):
            xi = x[i]
            k_row = []
            for s in self.support:
                k_val = self.kernel(xi, s)
                k_row.append(k_val)
            kernel_features.append(torch.cat(k_row))
        kernel_features = torch.stack(kernel_features)
        logits = self.linear(kernel_features)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuanvolutionFilter", "Kernel", "HybridQuantumBinaryClassifier"]
