"""Hybrid classical implementation of QCNN + Quanvolution + Sampler."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------  Classical Quanvolution Filter -------------
class ClassicalQuanvolutionFilter(nn.Module):
    """
    Mimics the patch‑wise quantum kernel by a lightweight 2×2 convolution.
    The architecture can be replaced with the true quantum filter when a
    backend is available.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x shape: (batch, 1, 28, 28)
        return self.conv(x).view(x.size(0), -1)  # flatten patches

# -------------  Classical QCNN backbone -------------
class ClassicalQCNNBackbone(nn.Module):
    """
    Fully‑connected stack emulating the QCNN convolution–pooling sequence.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
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
        return self.conv3(x)

# -------------  Classical Sampler -------------
class ClassicalSampler(nn.Module):
    """
    Softmax classifier that mimics the quantum SamplerQNN.
    """
    def __init__(self, in_features: int, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 4),
            nn.Tanh(),
            nn.Linear(4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.log_softmax(self.net(x), dim=-1)

# -------------  Hybrid QCNN -------------
class HybridQCNN(nn.Module):
    """
    End‑to‑end classical proxy for the hybrid QCNN.  It chains:
      * QuanvolutionFilter  →  QCNNBackbone  →  Sampler
    The module can be trained end‑to‑end with standard optimizers.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter()
        self.backbone = ClassicalQCNNBackbone()
        # The output dimension of the backbone is 4; the sampler expands to num_classes
        self.classifier = ClassicalSampler(in_features=4, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.qfilter(x)
        x = self.backbone(x)
        return self.classifier(x)

def HybridQCNNFactory(num_classes: int = 10) -> HybridQCNN:
    """Return a ready‑to‑train HybridQCNN instance."""
    return HybridQCNN(num_classes=num_classes)

__all__ = ["HybridQCNNFactory", "HybridQCNN"]
