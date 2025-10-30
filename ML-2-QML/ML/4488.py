from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalEncoder(nn.Module):
    """Simple linear encoder that mimics a quantum data‑encoding layer."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ClassicalSelfAttention(nn.Module):
    """Classic self‑attention block mirroring the SelfAttention seed."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.q_proj.out_features), dim=-1)
        return scores @ v

class ClassicalQuanvolution(nn.Module):
    """Convolutional filter that keeps the classical interface of the original quanvolution."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class ClassicalQLSTM(nn.Module):
    """Classical LSTM that can replace the quantum LSTM in the hybrid model."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out

class HybridQuantumClassifier(nn.Module):
    """
    Unified model that stitches together:
    1.  data‑encoding (classical linear)
    2.  a self‑attention block
    3.  a quanvolution filter
    4.  a classical LSTM (placeholder for a quantum LSTM)
    5.  a final classical head.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        attention_embed: int = 4,
        quanv_out_channels: int = 4,
        classifier_depth: int = 2,
    ):
        super().__init__()
        self.encoder = ClassicalEncoder(input_dim)
        self.attention = ClassicalSelfAttention(attention_embed)
        self.qfilter = ClassicalQuanvolution(out_channels=quanv_out_channels)
        self.lstm = ClassicalQLSTM(input_dim=seq_len, hidden_dim=hidden_dim)
        classifier_layers = []
        in_features = hidden_dim
        for i in range(classifier_depth - 1):
            classifier_layers.append(nn.Linear(in_features, 32))
            classifier_layers.append(nn.ReLU())
            in_features = 32
        classifier_layers.append(nn.Linear(in_features, 2))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: batch x seq_len x features
        encoded = self.encoder(x)
        attn = self.attention(encoded)
        batch = x.size(0)
        # placeholder reshape to a 28x28 image for the quanvolution
        img = attn.view(batch, 1, 28, 28)
        quanv = self.qfilter(img)
        lstm_input = quanv.unsqueeze(1)  # batch x 1 x features
        lstm_out = self.lstm(lstm_input)
        out = self.classifier(lstm_out[:, -1, :])
        return out

__all__ = ["HybridQuantumClassifier"]
