from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

class QuanvolutionFilter(nn.Module):
    """2×2 patch extractor with a learnable linear projection."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        return self.conv(x).view(x.size(0), -1)

class QCNNModel(nn.Module):
    """Classical emulation of a quantum convolutional network."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

@dataclass
class FraudLayerParameters:
    """Parameters that map to both classical and quantum layers."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model that mirrors the photonic circuit with a
    classical analogue: a Quanvolution feature extractor, a QCNN‑style
    feature map, and a linear regressor that mimics the final photonic
    displacement layer.
    """
    def __init__(self, params: FraudLayerParameters) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.qcnn = QCNNModel()
        # Linear head replicates the final photonic displacement/phase shift
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)
        self.scale = nn.Parameter(scale)
        self.shift = nn.Parameter(shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract 2×2 patches and flatten
        features = self.qfilter(x)
        # Pass through QCNN feature extractor
        features = self.qcnn(features)
        # Apply learned scaling and shift (analogous to displacement)
        return features * self.scale + self.shift

    def load_photonic_params(self, params: FraudLayerParameters) -> None:
        """Map photonic parameters to the classical model."""
        self.scale.data.copy_(torch.tensor(params.displacement_r))
        self.shift.data.copy_(torch.tensor(params.displacement_phi))

__all__ = ["FraudDetectionHybrid", "FraudLayerParameters"]
