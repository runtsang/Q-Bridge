"""Hybrid classical classifier with convolution‑inspired architecture.

The class mirrors the quantum helper interface and extends the
classical feed‑forward factory by embedding a QCNN‑style feature
extraction pipeline.  The network can be used as a stand‑alone
classifier or as a feature extractor for a downstream quantum layer.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple

class HybridQuantumClassifier(nn.Module):
    """Convolution‑inspired feed‑forward network with optional depth control.

    The architecture follows the pattern of the QCNNModel but replaces
    the fully‑connected layers with a depth‑parameterised stack of
    linear + activation blocks.  The final head outputs logits for a
    binary classification task.
    """

    def __init__(self, input_dim: int = 8, depth: int = 3) -> None:
        super().__init__()
        # Feature map – first linear layer
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())

        # Convolution‑like blocks
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Optional additional depth layers
        self.additional = nn.ModuleList()
        for _ in range(depth - 3):
            self.additional.append(nn.Sequential(nn.Linear(4, 4), nn.Tanh()))

        # Classification head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        for layer in self.additional:
            x = layer(x)
        return torch.sigmoid(self.head(x))

def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Return a hybrid network and metadata for compatibility with the
    quantum helper interface.

    The returned tuple follows the signature used by the original
    QuantumClassifierModel.py: ``(network, encoding, weight_sizes,
    observables)``.  ``encoding`` and ``observables`` are placeholders
    that allow the quantum code to treat the classical network as a
    feature extractor.
    """
    network = HybridQuantumClassifier(num_features, depth)
    # Encoding indices correspond to the input feature positions
    encoding = list(range(num_features))
    # Compute number of trainable parameters per layer
    weight_sizes = []
    for module in network.modules():
        if isinstance(module, nn.Linear):
            weight_sizes.append(module.weight.numel() + module.bias.numel())
    # Observables placeholder – binary classification
    observables = [0, 1]
    return network, encoding, weight_sizes, observables

__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]
