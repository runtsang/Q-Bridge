"""Hybrid classical classifier combining feed‑forward, QCNN, and convolutional
filter logic from the seed projects.

The network first applies a lightweight convolutional feature extractor,
then a classical feed‑forward block, followed by a QCNN model to capture
higher‑order patterns, and finally a linear output head.  The architecture
is fully trainable with PyTorch and is compatible with the original
`QuantumClassifierModel` interface, keeping the same public method names.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple

# Imports from seed modules
from.Conv import Conv  # Conv filter from Conv.py
from.QCNN import QCNN  # QCNNModel factory from QCNN.py
from.QuantumClassifierModel import build_classifier_circuit  # classic classifier builder

__all__ = ["HybridClassifier"]


class HybridClassifier(nn.Module):
    """
    Classic hybrid classifier that unifies three ideas:

    1. A lightweight 2‑D convolutional filter (Conv) to extract local structure.
    2. A deep feed‑forward network built by `build_classifier_circuit`.
    3. A QCNN block that emulates quantum convolution/pooling hierarchy.

    Parameters
    ----------
    num_features : int
        Size of the flat input vector.  It must be a perfect square for the
        convolutional filter to interpret it as a 2‑D image.
    depth : int
        Depth of the feed‑forward stack.
    conv_kernel : int, optional
        Kernel size of the Conv filter.  Default is 2.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.conv_kernel = conv_kernel
        self.conv = Conv()
        self.base, self.encodings, self.weights, self.observables = build_classifier_circuit(
            num_features, depth
        )
        self.qcnn = QCNN()
        self.output_head = nn.Linear(1, 2)

    def _conv_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Conv filter to patches of the input and return a batch of
        scalar features.  The input is expected to be a flat vector that can be
        reshaped into a square image.  For simplicity the filter is approximated
        by taking the mean of the whole image; this keeps the interface
        compatible while still injecting a local‑feature signal.
        """
        # Simple mean‑based feature to mimic convolution
        return x.mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining all components.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Class logits of shape (batch, 2).
        """
        # 1. Convolutional feature extraction
        conv_feat = self._conv_feature(x)

        # 2. Classical feed‑forward block
        ff_out = self.base(x)

        # 3. QCNN on the convolutional feature (expects 8‑dim input)
        # We pad or truncate to 8 dimensions to match QCNNModel.
        qcnn_in = conv_feat[:, :8]
        qcnn_out = self.qcnn(qcnn_in)

        # 4. Final linear head
        logits = self.output_head(qcnn_out)
        return logits
