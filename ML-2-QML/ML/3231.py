"""Hybrid classical classifier combining a QCNN feature extractor with a feed‑forward head."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn

# Import the pure‑classical QCNN model from the seed
try:
    from QCNN import QCNNModel
except Exception:  # pragma: no cover
    QCNNModel = None  # placeholder if QCNN module is missing


class HybridClassifierModel(nn.Module):
    """Classifier that optionally prepends a QCNN feature extractor."""

    def __init__(self, num_features: int, depth: int, use_qcnn: bool = False, qcnn_features: int = 8) -> None:
        super().__init__()
        self.use_qcnn = use_qcnn
        self.qcnn_features = qcnn_features

        if self.use_qcnn and QCNNModel is not None:
            self.feature_extractor = QCNNModel()
            in_dim = qcnn_features
        else:
            self.feature_extractor = None
            in_dim = num_features

        layers: List[nn.Module] = []
        current_dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(current_dim, num_features))
            layers.append(nn.ReLU())
            current_dim = num_features
        layers.append(nn.Linear(current_dim, 2))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        return self.classifier(x)


def build_classifier_network(
    num_features: int, depth: int, use_qcnn: bool = False, qcnn_features: int = 8
) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Factory mirroring the quantum helper interface.

    Returns:
        - model: the constructed :class:`HybridClassifierModel`.
        - encoding: indices of the input features that reach the network.
        - weight_sizes: number of trainable parameters per layer.
        - observables: dummy output class indices for downstream QNN wrappers.
    """
    model = HybridClassifierModel(num_features, depth, use_qcnn, qcnn_features)

    # encoding: simply all input feature indices
    encoding = list(range(num_features))

    # weight_sizes: count parameters in each linear layer
    weight_sizes = [p.numel() for p in model.parameters() if p.requires_grad]

    # observables: two output classes
    observables = [0, 1]

    return model, encoding, weight_sizes, observables


__all__ = ["HybridClassifierModel", "build_classifier_network"]
