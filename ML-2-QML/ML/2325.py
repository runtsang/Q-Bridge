"""Combined classical QCNN with sampler-inspired architecture.

This module defines the QCNNGen129 model that merges the convolutional
stack from QCNN.py with a lightweight sampler network inspired by SamplerQNN.py.
The architecture is fully classical but retains the same interface as the
original QCNN factory, allowing seamless replacement in downstream pipelines.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class QCNNGen129(nn.Module):
    """
    A fully connected network that emulates the QCNN convolutional and pooling
    layers, followed by a sampler-style softmax head.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature map: 8 → 16
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        # Convolution + pooling stages
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.pool3 = nn.Sequential(nn.Linear(4, 2), nn.Tanh())
        # Sampler-inspired head: softmax over 2 outputs
        self.sampler_head = nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        # Softmax sampler output
        return F.softmax(self.sampler_head(x), dim=-1)

def QCNN() -> QCNNGen129:
    """Factory that returns a fully connected QCNNGen129 instance."""
    return QCNNGen129()

__all__ = ["QCNN", "QCNNGen129"]
