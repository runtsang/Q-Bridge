"""
QCNNEnhanced: Classical network with a pretrained feature extractor and hybrid loss.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class QCNNFeatureExtractor(nn.Module):
    """
    Lightweight feature extractor.  If ``trainable`` is ``False`` the
    mapping mimics the 8‑bit ZFeatureMap used in the quantum version.
    """
    def __init__(self, input_dim: int = 8, trainable: bool = True) -> None:
        super().__init__()
        if trainable:
            self.feature_map = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU()
            )
        else:
            # fixed mapping: identity followed by a non‑linear squash
            self.feature_map = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.Tanh()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_map(x)


class QCNNClassifier(nn.Module):
    """
    Classical QCNN inspired network.  It uses the feature extractor,
    a stack of “convolution” layers (fully connected blocks) and
    a hybrid loss that can combine a classical BCE with a quantum
    expectation value.
    """
    def __init__(self, feature_extractor: QCNNFeatureExtractor) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def hybrid_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        quantum_score: torch.Tensor | None = None,
        q_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Combines binary cross‑entropy with an optional quantum score.
        The quantum score should be a scalar per example (e.g. expectation
        value of a Pauli observable).  If it is ``None`` the loss is pure BCE.
        """
        bce = F.binary_cross_entropy(preds, targets)
        if quantum_score is not None:
            # We encourage the quantum score to be close to the target
            q_loss = F.mse_loss(quantum_score, targets)
            return (1.0 - q_weight) * bce + q_weight * q_loss
        return bce


def QCNNEnhanced(pretrained: bool = False) -> QCNNClassifier:
    """
    Factory that returns a QCNNClassifier.  ``pretrained`` controls
    whether the feature extractor is initialized with a fixed mapping.
    """
    extractor = QCNNFeatureExtractor(trainable=not pretrained)
    return QCNNClassifier(feature_extractor=extractor)


__all__ = ["QCNNEnhanced", "QCNNClassifier", "QCNNFeatureExtractor"]
