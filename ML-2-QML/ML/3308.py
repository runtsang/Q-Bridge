from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple

# Classical convolutional filter (adapted from Conv.py)
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# Feed‑forward classifier builder (adapted from ML seed)
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class QuantumClassifierModel:
    """
    Classical classifier that fuses a learnable convolutional feature extractor
    with a depth‑controlled feed‑forward network.  The interface mirrors the
    quantum variant so that the two implementations can be swapped
    experimentally.
    """
    def __init__(
        self,
        num_features: int,
        depth: int,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        self.conv_filter = ConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth
        )

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Run the model on a 2‑D input patch.  The convolutional filter produces
        a scalar feature that is fed into the feed‑forward network.

        Args:
            data: 2‑D array of shape (conv_kernel, conv_kernel).

        Returns:
            np.ndarray: Raw logits of shape (2,).
        """
        feature = self.conv_filter.run(data)
        x = torch.tensor([feature], dtype=torch.float32)
        logits = self.network(x)
        return logits.detach().cpu().numpy()
