import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np

class QCNNModel(nn.Module):
    """A lightweight QCNN‑inspired feature extractor.

    The architecture mirrors the classical QCNN but adds dropout after each
    non‑linear block to improve regularisation.  It accepts an 8‑dimensional
    input and outputs a 4‑dimensional feature vector that can be fed into a
    downstream sampler.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(), nn.Dropout(p=0.1)
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16), nn.Tanh(), nn.Dropout(p=0.1)
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12), nn.Tanh(), nn.Dropout(p=0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8), nn.Tanh(), nn.Dropout(p=0.1)
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4), nn.Tanh(), nn.Dropout(p=0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4), nn.Tanh(), nn.Dropout(p=0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x

class SamplerModule(nn.Module):
    """Multi‑layer softmax sampler.

    Two hidden layers with ReLU activations map the extracted features
    to a probability distribution over two classes.  The final layer
    uses softmax so that the output can be interpreted as a categorical
    distribution.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(x), dim=-1)

class HybridSampler(nn.Module):
    """Combines a QCNN feature extractor with a classical sampler.

    The forward pass is a two‑stage pipeline: the input is first embedded
    by :class:`QCNNModel`, then the resulting features are passed to
    :class:`SamplerModule`.  The design mirrors the hybrid quantum‑classical
    workflow where a quantum ansatz extracts features and a classical
    network produces the final output.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = QCNNModel()
        self.sampler = SamplerModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.feature_extractor(x)
        return self.sampler(features)

def QCNN() -> QCNNModel:
    """Return a fresh QCNN feature extractor."""
    return QCNNModel()

def SamplerQNN() -> SamplerModule:
    """Return a fresh classical sampler module."""
    return SamplerModule()

__all__ = ["QCNN", "SamplerQNN", "HybridSampler"]
