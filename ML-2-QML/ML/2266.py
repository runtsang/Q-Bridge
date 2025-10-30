"""Hybrid classical sampler network combining convolutional structure from QCNN and sampler output from SamplerQNN.

The network first extracts features through fully‑connected layers that mimic the quantum
convolution and pooling steps of a QCNN. The final 4‑dimensional representation is then
mapped to a 2‑class probability vector using a softmax head, mirroring the SamplerQNN
output. This hybrid design leverages the expressive power of convolutional feature
extraction while retaining the simplicity of a classical sampler.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerQNN(nn.Module):
    """Hybrid classical sampler network.

    Architecture:
    - Feature extractor: 5 fully‑connected layers with Tanh activations,
      inspired by the QCNN convolution and pooling stages.
    - Sampler head: 2‑output softmax, inspired by the original SamplerQNN.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extraction mimicking QCNN
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Sampler head
        self.head = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return F.softmax(self.head(x), dim=-1)


def HybridSamplerQNN_factory() -> HybridSamplerQNN:
    """Factory returning an instance of :class:`HybridSamplerQNN`."""
    return HybridSamplerQNN()


__all__ = ["HybridSamplerQNN"]
