"""Combined classical convolution and classifier module.

The ConvGen212 class encapsulates a 2D convolution filter and a feed‑forward classifier.
It can be used as a drop‑in replacement for the original Conv filter while also providing
classification capabilities. The design follows the structure of the original Conv and
QuantumClassifierModel seeds, merging their strengths into a single, scalable interface.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple, List


class ConvGen212(nn.Module):
    """Convolution + classifier hybrid model.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel.
    threshold : float
        Threshold for activation after convolution.
    num_features : int
        Number of features for the classifier network.
    depth : int
        Number of hidden layers in the classifier.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        num_features: int = 10,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Build classifier
        self.classifier = self._build_classifier(num_features, depth)

        # Metadata similar to the quantum variant
        self.encoding = list(range(num_features))
        self.weight_sizes = self._compute_weight_sizes(self.classifier)
        self.observables = list(range(2))  # placeholder for classification outputs

    def _build_classifier(self, num_features: int, depth: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = 1  # classifier accepts a single scalar feature
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def _compute_weight_sizes(self, network: nn.Sequential) -> List[int]:
        sizes: List[int] = []
        for module in network:
            if isinstance(module, nn.Linear):
                sizes.append(module.weight.numel() + module.bias.numel())
        return sizes

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run the convolution filter and return the mean activation."""
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    def classify(self, data: torch.Tensor) -> torch.Tensor:
        """Classify the input data after convolution."""
        conv_out = self.forward(data)
        # Feed the scalar output into the classifier
        flat = conv_out.unsqueeze(0)
        return self.classifier(flat)

    def run(self, data) -> float:
        """Compatibility wrapper that accepts numpy arrays."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        return self.forward(tensor).item()


def Conv() -> ConvGen212:
    """Return a ConvGen212 instance as a drop‑in replacement."""
    return ConvGen212()


__all__ = ["ConvGen212", "Conv"]
