"""HybridQCNNQuanvolution: classical approximation of the QCNN + quanvolution hybrid network."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch extractor mimicking the quantum quanvolution kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QCNNFeatureExtractor(nn.Module):
    """Feature extractor that emulates the QCNN convolution and pooling sequence."""
    def __init__(self, in_features: int = 784) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(in_features, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x


class HybridQCNNQuanvolution(nn.Module):
    """Hybrid classical model combining quanvolution filter with QCNN feature extractor."""
    def __init__(self, in_features: int = 784, num_classes: int = 10) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        self.qcnn = QCNNFeatureExtractor(in_features=in_features)
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Classical quanvolution feature extraction
        features = self.quanvolution(x)
        # QCNN-inspired feature transformation
        features = self.qcnn(features)
        # Final classification head
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QCNNFeatureExtractor", "HybridQCNNQuanvolution"]
