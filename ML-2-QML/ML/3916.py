from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Tuple

class ConvFilter(nn.Module):
    """Emulates a quantum convolutional filter using classical 2D convolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data shape (batch, features) assumed square
        size = int(data.size(1) ** 0.5)
        data = data.view(data.size(0), 1, size, size)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])

class HybridConvClassifier(nn.Module):
    """Classical hybrid classifier: quantum‑inspired convolution + feed‑forward network."""
    def __init__(self,
                 num_features: int,
                 depth: int,
                 kernel_size: int = 2,
                 conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=kernel_size, threshold=conv_threshold)
        feat = (num_features - kernel_size + 1) ** 2
        layers = []
        in_dim = feat
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, feat))
            layers.append(nn.ReLU())
            in_dim = feat
        layers.append(nn.Linear(in_dim, 2))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

def build_classifier_circuit(num_features: int,
                             depth: int,
                             kernel_size: int = 2,
                             conv_threshold: float = 0.0) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Return a hybrid model together with metadata for compatibility."""
    model = HybridConvClassifier(num_features, depth, kernel_size, conv_threshold)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables = [0, 1]
    return model, encoding, weight_sizes, observables

__all__ = ["HybridConvClassifier", "build_classifier_circuit"]
