"""Hybrid classifier combining a CNN backbone with a quantum-enriched LSTM for sequence classification.

The module defines:
- `HybridConvNet`: a lightweight CNN that extracts features from images.
- `QuantumLSTMCellWrapper`: a wrapper around a quantum LSTM cell defined in the QML module.
- `HybridClassifierQLSTM`: the end‑to‑end model that processes sequences of images and outputs class probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import quantum LSTM cell wrapper from the QML module
from qml_module import QuantumLSTMCellWrapper

class HybridConvNet(nn.Module):
    """CNN backbone that processes individual images into feature vectors."""
    def __init__(self, in_channels: int = 3, conv_channels: list[int] | None = None):
        super().__init__()
        if conv_channels is None:
            conv_channels = [6, 15, 15]
        # conv1
        self.conv1 = nn.Conv2d(in_channels, conv_channels[0], kernel_size=5, stride=2, padding=1)
        # conv2
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1)
        # conv3
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout2d(p=0.4)
        # compute flatten size
        dummy = torch.zeros(1, in_channels, 64, 64)
        x = self.features(dummy)
        self.flat_dim = x.view(1, -1).size(1)
        # final linear to reduce to hidden_dim
        self.fc = nn.Linear(self.flat_dim, 128)

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

class HybridClassifierQLSTM(nn.Module):
    """Hybrid classifier that processes sequences of images using a CNN backbone and a quantum LSTM."""
    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, num_classes: int = 2, n_qubits: int = 4):
        super().__init__()
        self.backbone = HybridConvNet(in_channels)
        self.lstm = QuantumLSTMCellWrapper(hidden_dim, n_qubits)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_batch: torch.Tensor):
        """
        seq_batch: Tensor of shape (seq_len, batch, C, H, W)
        """
        seq_len, batch_size, C, H, W = seq_batch.shape
        features = []
        for t in range(seq_len):
            x_t = seq_batch[t]
            feat_t = self.backbone(x_t)  # (batch, hidden_dim)
            features.append(feat_t)
        features = torch.stack(features, dim=0)  # (seq_len, batch, hidden_dim)
        lstm_out, _ = self.lstm(features)  # (seq_len, batch, hidden_dim)
        # use last time step for classification
        logits = self.classifier(lstm_out[-1])
        probs = self.softmax(logits)
        return probs

__all__ = ["HybridConvNet", "HybridClassifierQLSTM", "QuantumLSTMCellWrapper"]
