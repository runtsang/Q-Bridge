"""Hybrid Quanvolution network combining classical self‑attention and a QCNN‑style stack."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalSelfAttention(nn.Module):
    """Self‑attention block inspired by the SelfAttention seed.

    The parameters are trainable torch tensors and are applied in the forward
    pass via pure torch operations to keep the interface identical to the seed.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply scaled dot‑product self‑attention."""
        query = torch.matmul(inputs, self.rotation_params)
        key = torch.matmul(inputs, self.entangle_params)
        value = inputs
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value


class QCNNModel(nn.Module):
    """QCNN‑style fully‑connected network adapted to produce 4‑dimensional output."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(4, 8), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(8, 6), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(6, 4), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(4, 3), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(3, 3), nn.Tanh())
        self.head = nn.Linear(3, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class HybridQuanvolutionNet(nn.Module):
    """Hybrid classical network that fuses a Conv‑like filter, self‑attention,
    and a QCNN‑style stack before classification."""

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.compress = nn.Linear(4 * 14 * 14, 4)
        self.attention = ClassicalSelfAttention(embed_dim=4)
        self.qcnn = QCNNModel()
        self.classifier = nn.Linear(4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv feature extraction
        conv_out = self.conv(x)                    # (batch, 4, 14, 14)
        conv_out = conv_out.view(x.size(0), -1)    # (batch, 4*14*14)
        conv_out = self.compress(conv_out)         # (batch, 4)

        # Attention over the compressed features
        attn_out = self.attention(conv_out)

        # QCNN processing
        qcnn_out = self.qcnn(attn_out)

        # Classification
        logits = self.classifier(qcnn_out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolutionNet"]
