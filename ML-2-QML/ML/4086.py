"""Hybrid sampler network combining classical and quantum inspired modules.

The module exposes a :class:`SamplerQNNHybrid` that:
  * runs a 2‑to‑4 soft‑max sampler (classical counterpart to the quantum sampler)
  * passes the concatenated 2‑feature input through a QCNN‑style
    multilayer perceptron
  * optionally augments the representation with a lightweight
    fully‑connected quantum‑layer emulation (FCL)
  * outputs a single sigmoid probability

The design mirrors the three reference seeds:
  * SamplerQNN.py – the soft‑max sampler
  * QCNN.py   – the convolutional style MLP
  * FCL.py    – a quantum‑layer emulation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable


class FullyConnectedLayer(nn.Module):
    """Lightweight emulation of a quantum fully‑connected layer.

    Implements the same interface as the original FCL class: a ``run``
    method that accepts a list of angles and returns a single value
    computed from a tanh activation.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation


class QCNNFeatureExtractor(nn.Module):
    """QCNN‑style feature extractor using fully connected layers."""
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
        return self.head(x)


class SamplerQNNHybrid(nn.Module):
    """Hybrid sampler integrating classical sampler, QCNN extractor and FCL."""
    def __init__(self) -> None:
        super().__init__()
        # Classical sampler
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # QCNN feature extractor
        self.qcnn = QCNNFeatureExtractor()
        # FCL emulation
        self.fcl = FullyConnectedLayer()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Batch of 2‑dimensional feature vectors.

        Returns
        -------
        torch.Tensor
            Batch of sigmoid probabilities.
        """
        # Classical sampler output
        sampler_out = F.softmax(self.sampler(inputs), dim=-1)

        # Concatenate sampler output with original features
        combined = torch.cat([inputs, sampler_out], dim=-1)  # shape: (batch, 4)

        # Pad to 8 dimensions for the QCNN extractor
        padded = torch.nn.functional.pad(combined, (0, 4), mode='constant')

        # QCNN feature extraction
        qcnn_out = self.qcnn(padded)

        # Additional quantum‑layer emulation
        fcl_out = self.fcl.run(padded.squeeze(0).tolist()).unsqueeze(0)

        # Combine QCNN and FCL signals
        logits = qcnn_out + fcl_out

        return torch.sigmoid(logits)


def SamplerQNN() -> SamplerQNNHybrid:
    """Factory returning the hybrid sampler network."""
    return SamplerQNNHybrid()


__all__ = ["SamplerQNNHybrid", "SamplerQNN"]
