"""Hybrid quantum‑classical binary classifier for image data.

The module provides a pure‑classical implementation that mirrors the
quantum‑enhanced architecture from the QML seed.  All quantum components
are replaced by dense layers and a classical LSTM.  This class can be
instantiated and trained on CPU/GPU without any quantum backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Classical LSTM replacement
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """Drop‑in classical LSTM that mimics the gate structure of the quantum LSTM."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1, :]  # return last hidden state

# --------------------------------------------------------------------------- #
# 2. Classical head
# --------------------------------------------------------------------------- #
class ClassicalHead(nn.Module):
    """Linear + sigmoid head producing a 2‑class probability vector."""
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        prob = torch.sigmoid(logits)
        return torch.cat([prob, 1 - prob], dim=1)

# --------------------------------------------------------------------------- #
# 3. CNN Backbone
# --------------------------------------------------------------------------- #
class CNNBackbone(nn.Module):
    """Convolutional feature extractor that outputs a sequence of scalars."""
    def __init__(self, seq_len: int = 50) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, seq_len)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # (batch, seq_len)

# --------------------------------------------------------------------------- #
# 4. Full hybrid classifier (classical)
# --------------------------------------------------------------------------- #
class HybridQuantumBinaryClassifier(nn.Module):
    """Pure‑classical hybrid model that can later be swapped with the quantum
    implementation.  The architecture mirrors the original QML seed:
    CNN → LSTM → linear head.
    """
    def __init__(self, seq_len: int = 50, lstm_hidden_dim: int = 32) -> None:
        super().__init__()
        self.backbone = CNNBackbone(seq_len=seq_len)
        self.lstm = ClassicalQLSTM(input_dim=1, hidden_dim=lstm_hidden_dim)
        self.head = ClassicalHead(in_features=lstm_hidden_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3, H, W)
        features = self.backbone(x)          # (batch, seq_len)
        seq = features.unsqueeze(-1)        # (batch, seq_len, 1)
        lstm_out = self.lstm(seq)           # (batch, hidden_dim)
        probs = self.head(lstm_out)         # (batch, 2)
        return probs

__all__ = ["HybridQuantumBinaryClassifier"]
