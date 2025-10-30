"""Classical hybrid fraud detection model combining convolutional, LSTM, and dense head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQuanvolutionFilter(nn.Module):
    """Simple 2×2 convolution followed by flattening, mimicking the quantum filter."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class ClassicalLSTMTagger(nn.Module):
    """Sequence model using a standard LSTM."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)
        out = self.fc(out[:, -1, :])  # use last hidden state
        return torch.sigmoid(out)

class FraudDetectionHybridModel(nn.Module):
    """
    Classical analogue of the hybrid fraud detection architecture.
    Combines a classical convolutional feature extractor, a
    sequential LSTM, and a dense classification head.
    """
    def __init__(self, in_channels: int = 1, seq_len: int = 1):
        super().__init__()
        self.feature_extractor = ClassicalQuanvolutionFilter(in_channels)
        # The flattened feature dimension is 4 * 14 * 14 for 28x28 input
        self.sequence_model = ClassicalLSTMTagger(input_dim=4 * 14 * 14, hidden_dim=128)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        features = self.feature_extractor(x)
        # Reshape to (batch, seq_len, feature_dim)
        seq = features.unsqueeze(1)  # seq_len=1 for static input
        lstm_out = self.sequence_model(seq)
        logits = self.classifier(lstm_out)
        return torch.sigmoid(logits)

__all__ = ["FraudDetectionHybridModel"]
