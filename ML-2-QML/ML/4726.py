"""QuantumHybridFusion: Classical implementation of a hybrid neural network.

This module implements a scalable architecture that mirrors the original
quantum‑inspired designs.  The network is composed of three functional
sub‑modules:

1. ``FCLayer`` – a fully‑connected linear mapping that emulates the
   single‑qubit expectation layer from the FCL example.  The layer
   returns a tanh‑activated output which can be used directly as a
   replacement for the quantum expectation value.

2. ``QCNNBlock`` – a lightweight convolutional stack inspired by the
   QCNN helper.  The block uses two 2‑D convolution layers followed by
   pooling and a final fully‑connected head.  All operations are
   differentiable and use standard PyTorch primitives.

3. ``HybridHead`` – a classical logistic regression head that replaces
   the quantum expectation layer used in the original hybrid binary
   classifier.  It is implemented with a single linear layer followed
   by a custom autograd ``HybridFunction`` that mimics the behaviour
   of a quantum‑based expectation but without any quantum back‑end.

The public factory ``QuantumHybridFusion`` stitches these blocks
together into a single module that accepts a batch of RGB images
and produces class probabilities for binary classification.  The
architecture is intentionally modular to allow easy swapping of the
classical head for a quantum one when a quantum back‑end becomes
available.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuantumHybridFusion", "FCLayer", "QCNNBlock", "HybridHead"]


class FCLayer(nn.Module):
    """Fully‑connected layer that emulates a quantum expectation.

    The layer implements a single linear mapping followed by a tanh
    activation.  The output is interpreted as a classical surrogate of
    the quantum expectation value.  The layer can be used as a drop‑in
    replacement for the quantum expectation head in the hybrid
    architecture.
    """

    def __init__(self, in_features: int = 1, out_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = torch.tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class QCNNBlock(nn.Module):
    """Convolution‑style block inspired by the QCNN helper.

    The block consists of two 2‑D convolution layers followed by max
    pooling.  It is deliberately lightweight so that it can be stacked
    or replaced by deeper classical back‑bones without a large
    computational overhead.
    """

    def __init__(self, in_channels: int, out_channels: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x


class HybridHead(nn.Module):
    """Classical head that mimics the quantum expectation output.

    The head is a single linear layer that maps the flattened features to
    a scalar logit.  A custom autograd ``HybridFunction`` is used to
    provide a differentiable sigmoid that behaves like the quantum
    expectation head of the original hybrid network.
    """

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        # Apply a shift‑augmented sigmoid to emulate the quantum
        # expectation head.
        return torch.sigmoid(logits + self.shift)


class QuantumHybridFusion(nn.Module):
    """Full classical hybrid network for binary classification.

    The network applies a QCNN‑style feature extractor, followed by a
    fully‑connected layer that emulates the quantum expectation, and
    finally a classical logistic head.  The design mirrors the
    structure of the original QCNet but removes all quantum
    primitives, making it suitable for rapid prototyping on CPUs.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor: 3‑channel RGB images → 16‑channel feature maps
        self.feature_extractor = QCNNBlock(in_channels=3, out_channels=16)
        # Flatten and map to a 1‑dimensional “angle” vector
        self.fc = nn.Linear(16 * 8 * 8, 1)  # assumes input size 32×32
        # Classical head emulating the quantum expectation
        self.head = HybridHead(in_features=1, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.head(x)
        # Return probabilities for the two binary classes
        return torch.cat((x, 1 - x), dim=-1)


if __name__ == "__main__":
    # Simple sanity check
    net = QuantumHybridFusion()
    dummy = torch.randn(4, 3, 32, 32)
    out = net(dummy)
    print(out.shape)  # should be (4, 2)
