"""HybridQCNN: classical backbone + quantum convolutional backbone + hybrid head.

The implementation is a drop‑in replacement for the original QCNN.py
and extends the model by incorporating
  * a small dense feature‑map network that feeds into both branches,
  * a quantum convolutional stack that mirrors the classical
    convolution‑pooling pattern,
  * **the hybrid head** (QuantumExpectation) that takes both
    dense features and the quantum feature vector and returns
    a binary classification probability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Classical dense feature extractor
# --------------------------------------------------------------------------- #
class FeatureMapML(nn.Module):
    """A small dense network that produces a shared feature map."""
    def __init__(self, in_features: int = 8, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
        )
        self.out = nn.Linear(hidden, 1)  # One‑to‑one mapping for simplicity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(x))


# --------------------------------------------------------------------------- #
# 2. QCNNModel (quantum‑like fully‑connected stack)
# --------------------------------------------------------------------------- #
class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating the quantum convolution steps."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


# --------------------------------------------------------------------------- #
# 3. HybridQCNN: classical + quantum‑like + hybrid head
# --------------------------------------------------------------------------- #
class HybridQCNN(nn.Module):
    """Hybrid convolutional network with a classical backbone, a quantum‑like stack, and a hybrid head."""
    def __init__(self) -> None:
        super().__init__()
        # Shared feature map
        self.feature_map = FeatureMapML()

        # Classical convolutional backbone
        self.classical = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )

        # Quantum‑like fully‑connected stack (mimicking QCNN)
        self.quan_net = nn.Sequential(
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh(),
        )
        self.quan_head = nn.Linear(4, 1)

        # Hybrid head
        self.head = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical branch
        c_out = self.classical(x)  # shape [batch, 1]

        # Quantum‑like branch
        flat = x.view(x.size(0), -1)          # flatten to match QCNNModel input
        qm_feat = self.feature_map(flat)      # shape [batch, 1]
        qm_in = qm_feat.repeat(1, 16)         # shape [batch, 16]
        qm_out = self.quan_net(qm_in)         # shape [batch, 4]
        qm_out = self.quan_head(qm_out)       # shape [batch, 1]

        # Combine
        combined = torch.cat((c_out, qm_out), dim=-1)  # [batch, 2]
        logits = self.head(combined)
        probs = torch.sigmoid(logits)
        return probs


__all__ = ["HybridQCNN", "FeatureMapML", "QCNNModel"]
