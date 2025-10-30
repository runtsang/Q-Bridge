"""Hybrid classical estimator that blends feed‑forward, convolution‑style, and sampler logic.

The network first extracts features with a lightweight convolutional stack (QCNN‑style),
then optionally applies a softmax classifier (SamplerQNN‑style) or a regression head
(EstimatorQNN‑style).  The architecture is fully PyTorch‑based and can be used as a
stand‑alone regressor or classifier.
"""

import torch
from torch import nn
import torch.nn.functional as F


class HybridEstimatorQNN(nn.Module):
    """
    Classical hybrid estimator.

    Parameters
    ----------
    num_features : int, optional
        Number of input features (default 2).
    num_classes : int, optional
        If >1, the network outputs a probability distribution via softmax.
        If 1, the network outputs a single regression value.
    """

    def __init__(self, num_features: int = 2, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor – QCNN‑style
        self.feature_map = nn.Sequential(
            nn.Linear(num_features, 8), nn.Tanh()
        )
        self.conv1 = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Output head
        if self.num_classes == 1:
            self.head = nn.Linear(4, 1)
        else:
            self.head = nn.Linear(4, self.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        out = self.head(x)
        if self.num_classes > 1:
            out = F.softmax(out, dim=-1)
        return out


def HybridEstimatorQNNFactory(num_features: int = 2, num_classes: int = 1) -> HybridEstimatorQNN:
    """Factory returning a configured :class:`HybridEstimatorQNN`."""
    return HybridEstimatorQNN(num_features=num_features, num_classes=num_classes)


__all__ = ["HybridEstimatorQNN", "HybridEstimatorQNNFactory"]
