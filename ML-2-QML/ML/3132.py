from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerCNN(nn.Module):
    """
    A classical network that emulates the QCNN architecture and outputs a soft‑max
    distribution suitable for sampling.  The layer widths mirror the quantum
    ansatz used in the QML side, enabling parameter sharing during hybrid training.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extraction (analogous to the QCNN feature_map)
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        # Convolutional and pooling stages
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Output layer producing two logits
        self.head = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass that returns a probability distribution over two classes.
        """
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        logits = self.head(x)
        return F.softmax(logits, dim=-1)


def HybridSamplerCNN() -> HybridSamplerCNN:
    """
    Factory returning a freshly instantiated :class:`HybridSamplerCNN`.
    """
    return HybridSamplerCNN()


__all__ = ["HybridSamplerCNN", "HybridSamplerCNN"]
