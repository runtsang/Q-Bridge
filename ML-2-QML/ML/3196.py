"""Hybrid classifier providing classical, QCNN, and hybrid quantum‑classical modes.

The class exposes a uniform interface and re‑uses the feed‑forward design from
the original `QuantumClassifierModel`, the convolution‑pooling pattern of
`QCNN`, and a hybrid head that can consume a quantum feature vector.

Typical usage::

    model = HybridClassifier(mode="classical", num_features=8, depth=3)
    logits = model(torch.randn(1, 8))
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List

class HybridClassifier(nn.Module):
    """Combines classical feed‑forward, QCNN‑inspired, and hybrid quantum‑classical classifiers.

    Parameters
    ----------
    mode : {"classical", "qcnn", "hybrid"}
        The type of classifier to instantiate.
    num_features : int
        Number of input features (classical) or qubits (quantum).
    depth : int, default 3
        Depth of the feed‑forward layers or QCNN layers.
    """

    def __init__(self, mode: str, num_features: int, depth: int = 3) -> None:
        super().__init__()
        self.mode = mode
        if mode == "classical":
            self.model = self._build_classical(num_features, depth)
        elif mode == "qcnn":
            self.model = self._build_qcnn(num_features, depth)
        elif mode == "hybrid":
            self.model = self._build_hybrid(num_features, depth)
        else:
            raise ValueError(f"Unsupported mode {mode!r}")

    @staticmethod
    def _build_classical(num_features: int, depth: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_qcnn(num_features: int, depth: int) -> nn.Module:
        # Re‑use the QCNNModel defined in the original seed.
        class QCNNModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
                self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
                self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
                self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
                self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
                self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
                self.head = nn.Linear(4, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                x = self.feature_map(x)
                x = self.conv1(x)
                x = self.pool1(x)
                x = self.conv2(x)
                x = self.pool2(x)
                x = self.conv3(x)
                return torch.sigmoid(self.head(x))

        return QCNNModel()

    @staticmethod
    def _build_hybrid(num_features: int, depth: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
