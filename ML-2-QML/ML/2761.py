"""Hybrid classical convolution + classifier, drop‑in replacement for Conv.py.

The class combines a convolutional filter with a feed‑forward classifier,
mirroring the structure of the quantum counterparts from the seed projects.

Features
* Conv filter with learnable bias and threshold.
* Flexible depth classifier built with nn.Linear layers.
* API identical to the original Conv() factory.
"""

from __future__ import annotations

import torch
from torch import nn

def _build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """Return a sequential network and metadata matching the quantum version."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class HybridConvClassifier(nn.Module):
    """Classical hybrid of convolution filter and classifier."""
    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        num_features: int = 10,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        self.classifier, self.encoding, self.weight_sizes, self.observables = _build_classifier_circuit(
            num_features, depth
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply convolution, activation and classifier."""
        x = self.conv(data)
        x = torch.sigmoid(x - self.conv_threshold)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def run(self, data: torch.Tensor | list[list[float]]) -> torch.Tensor:
        """Convenience wrapper for numpy‑style input."""
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        data = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.forward(data)
        probs = torch.softmax(logits, dim=1)
        return probs

def Conv() -> HybridConvClassifier:
    """Factory that returns a HybridConvClassifier instance."""
    return HybridConvClassifier()

__all__ = ["HybridConvClassifier", "Conv"]
