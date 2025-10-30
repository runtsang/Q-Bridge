"""Hybrid quantum‑classical classifier mirroring the quantum helper interface.

The network mirrors the quantum helper interface: it returns a PyTorch model together with
* encoding indices – the positions of the classical features that will be mapped to qubits,
* weight sizes – number of trainable parameters per layer, and
* observables – indices of the output dimensions that correspond to quantum measurement outcomes.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn

class HybridClassifier(nn.Module):
    """
    Classical CNN + FC classifier with metadata compatible with a quantum circuit.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        depth: int = 2,
        feature_dim: int = 4,
    ) -> None:
        super().__init__()
        # Feature extractor (mirrors QFCModel.features)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten to a feature vector
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
        )
        # Feed‑forward head of configurable depth
        head_layers: List[nn.Module] = []
        in_dim = feature_dim
        for _ in range(depth):
            head_layers.append(nn.Linear(in_dim, feature_dim))
            head_layers.append(nn.ReLU())
            in_dim = feature_dim
        head_layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*head_layers)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        feature_vec = self.fc(flat)
        logits = self.head(feature_vec)
        return self.norm(logits)

def build_classifier_circuit(
    num_features: int,
    depth: int,
    feature_dim: int = 4,
    num_classes: int = 2,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a hybrid classifier model and return metadata.

    Parameters
    ----------
    num_features : int
        Number of raw input features (e.g., image channels).
    depth : int
        Depth of the feed‑forward head.
    feature_dim : int
        Size of the feature vector output by the CNN.
    num_classes : int
        Number of classification outputs.

    Returns
    -------
    network : nn.Module
        The constructed PyTorch model.
    encoding : Iterable[int]
        Indices of the feature vector that will be encoded into qubits.
    weight_sizes : Iterable[int]
        Number of parameters per layer (excluding bias).
    observables : List[int]
        Indices of the output logits that correspond to quantum measurement outcomes.
    """
    # Build network
    network = HybridClassifier(
        num_classes=num_classes, depth=depth, feature_dim=feature_dim
    )

    # Encoding indices: map the first `feature_dim` features to qubits
    encoding = list(range(feature_dim))

    # Compute weight sizes
    weight_sizes: List[int] = []
    for m in network.modules():
        if isinstance(m, nn.Linear):
            weight_sizes.append(m.weight.numel() + m.bias.numel())

    # Observables: map each class to an observable index
    observables = list(range(num_classes))
    return network, encoding, weight_sizes, observables

__all__ = ["HybridClassifier", "build_classifier_circuit"]
