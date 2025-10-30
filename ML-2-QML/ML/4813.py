"""Hybrid classical model integrating Quantum‑NAT CNN, QCNN‑style feed‑forward, and classifier head.

The network mirrors the structure of the original Quantum‑NAT CNN, the QCNN
feed‑forward pipeline, and the classifier construction from the seed
projects.  It can be trained end‑to‑end on classical data while exposing
interfaces that parallel the quantum counterpart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATHybrid(nn.Module):
    """
    Classical hybrid model.

    Architecture:
        1. 2‑layer CNN encoder (1 → 8 → 16 channels).
        2. QCNN‑style fully‑connected block:
           - feature_map  : Linear(16, 32)
           - conv1        : Linear(32, 32)
           - pool1        : Linear(32, 24)
           - conv2        : Linear(24, 16)
           - pool2        : Linear(16, 12)
           - conv3        : Linear(12, 12)
        3. Classifier head: Linear(12, num_classes).

    All intermediate layers use Tanh activations, mirroring the classical
    QCNN seed.  BatchNorm1d normalises the final 12‑dimensional feature
    vector, matching the normalisation used in the quantum seed.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        conv_features: int = 8,
        use_dropout: bool = False,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        # --- CNN encoder ----------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, conv_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_features, conv_features * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Feature map from the 16‑channel output (7×7) to 32 dimensions
        self.feature_map = nn.Linear(conv_features * 2 * 7 * 7, 32)

        # QCNN‑style block
        self.qcnn_block = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 24),
            nn.Tanh(),
            nn.Linear(24, 16),
            nn.Tanh(),
            nn.Linear(16, 12),
            nn.Tanh(),
            nn.Linear(12, 12),
            nn.Tanh(),
        )

        # Classifier head
        self.classifier = nn.Linear(12, num_classes)

        # Optional dropout
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_p)

        # Normalisation of the final feature vector
        self.norm = nn.BatchNorm1d(12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Class scores of shape (batch, num_classes).
        """
        # CNN encoder
        features = self.encoder(x)
        flat = features.view(features.size(0), -1)

        # Feature map
        fm = self.feature_map(flat)

        # QCNN‑style processing
        qcnn_out = self.qcnn_block(fm)

        # Optional dropout
        if self.use_dropout:
            qcnn_out = self.dropout(qcnn_out)

        # Normalise
        qcnn_out = self.norm(qcnn_out)

        # Classifier
        logits = self.classifier(qcnn_out)
        return logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the 12‑dimensional encoded feature vector before the
        classification head.  Useful for visualisation or as input to a
        downstream quantum circuit.
        """
        with torch.no_grad():
            features = self.encoder(x)
            flat = features.view(features.size(0), -1)
            fm = self.feature_map(flat)
            qcnn_out = self.qcnn_block(fm)
            if self.use_dropout:
                qcnn_out = self.dropout(qcnn_out)
            qcnn_out = self.norm(qcnn_out)
        return qcnn_out


__all__ = ["QuantumNATHybrid"]
