"""HybridQuanvolution – classical implementation with configurable depth and filter size.

The class exposes a convolutional patch extractor followed by a feed‑forward
classifier.  A static factory mirrors the quantum interface by returning a
PyTorch Sequential network together with metadata (encoding indices,
weight sizes and output classes).  The design allows easy comparison with
the quantum counterpart and supports hybrid training pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

def build_classifier_network(
    num_features: int,
    depth: int,
    num_classes: int = 10,
) -> Tuple[nn.Sequential, Iterable[int], List[int], List[int]]:
    """
    Construct a feed‑forward classifier mirroring the quantum variant.
    Returns the network, an encoding list, weight‑size list and output
    class indices.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, num_classes)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    outputs = list(range(num_classes))
    return network, encoding, weight_sizes, outputs

class HybridQuanvolution(nn.Module):
    """
    Classical analogue of the quanvolution architecture.

    Parameters
    ----------
    patch_size : int
        Size of the square patch to convolve over the image.
    depth : int
        Number of hidden layers in the linear head.
    num_features : int
        Number of output channels of the patch extractor.
    in_channels : int
        Number of input channels (default 1 for MNIST).
    num_classes : int
        Number of classification targets.
    """

    def __init__(
        self,
        patch_size: int = 2,
        depth: int = 1,
        num_features: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        stride = patch_size
        self.patch_conv = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, stride=stride
        )
        patch_grid = 28 // patch_size
        self.classifier, _, _, _ = build_classifier_network(
            num_features * patch_grid * patch_grid, depth, num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract patches, flatten, classify.
        Returns log‑softmax log‑probabilities.
        """
        features = self.patch_conv(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    @staticmethod
    def build_classifier_circuit(*_, **__) -> None:
        """
        Placeholder for API compatibility.  Quantum counterpart implements
        the circuit construction.
        """
        raise NotImplementedError(
            "Quantum circuit construction is available in the QML module."
        )

__all__ = ["HybridQuanvolution", "build_classifier_network"]
