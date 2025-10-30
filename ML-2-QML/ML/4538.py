"""Hybrid classical model combining convolution, quantum kernel, and fully‑connected layers.

The class ConvGen225 can be instantiated with optional hyper‑parameters and exposes a
`forward` method that accepts a batch of images.  It can be used as a drop‑in
replacement for the original Conv filter while providing additional
kernel‑based feature extraction and a fully‑connected head inspired by
Quantum‑NAT.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

# Classical convolution filter (from Conv.py)
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# RBF kernel module (from QuantumKernelMethod.py)
class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

# Fully‑connected head (from FCL.py)
class FullyConnectedLayer(nn.Module):
    def __init__(self, n_features: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: list[float]) -> np.ndarray:
        values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

# CNN backbone (from QuantumNAT.py)
class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

# Hybrid model
class ConvGen225(nn.Module):
    """Hybrid model that chains a classical convolution, an RBF kernel, and a fully‑connected head."""
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        gamma: float = 1.0,
        n_fc_features: int = 4,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size, threshold)
        self.kernel = Kernel(gamma)
        self.cnn = SimpleCNN()
        self.fc = nn.Linear(4 + 1, n_fc_features)  # 4 from CNN, 1 from kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical convolution on local patches
        conv_out = self.conv.run(x.detach().cpu().numpy())
        conv_tensor = torch.tensor([conv_out], dtype=torch.float32)

        # Kernel similarity with a learned prototype
        proto = torch.nn.Parameter(torch.randn_like(conv_tensor))
        kernel_val = self.kernel(conv_tensor, proto).unsqueeze(1)

        # CNN backbone features
        cnn_features = self.cnn(x)

        # Combine kernel and CNN features
        combined = torch.cat([cnn_features, kernel_val], dim=1)
        return self.fc(combined)

def Conv() -> ConvGen225:
    """Return a fully‑initialized hybrid model."""
    return ConvGen225()
