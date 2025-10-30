import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalQuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution acting as a coarse feature extractor."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)


class ClassicalSelfAttention(nn.Module):
    """A lightweight self‑attention wrapper compatible with the Qiskit API."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class ClassicalQLSTM(nn.Module):
    """Drop‑in classical LSTM that mirrors the quantum interface."""
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


def build_classifier(num_features: int, depth: int) -> nn.Sequential:
    """Construct a feed‑forward classifier with ReLU activations."""
    layers = []
    in_dim = num_features
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.ReLU())
        in_dim = num_features
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)


class QuanvolutionHybrid(nn.Module):
    """
    Classical hybrid model that chains:
      1. Convolutional patch extraction
      2. Optional self‑attention
      3. LSTM sequence modeling
      4. Fully‑connected classifier
    """
    def __init__(
        self,
        patch_dim: int = 4 * 14 * 14,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        classifier_depth: int = 2,
    ) -> None:
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        self.attention = ClassicalSelfAttention(embed_dim=4)
        self.lstm = ClassicalQLSTM(patch_dim, lstm_hidden, lstm_layers)
        self.classifier = build_classifier(lstm_hidden, classifier_depth)

        # Learned parameters for the attention block
        self.rotation = nn.Parameter(torch.randn(4, 4))
        self.entangle = nn.Parameter(torch.randn(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Convolutional feature extraction
        features = self.filter(x)  # shape: [B, patch_dim]

        # 2. Reshape into a sequence of patch vectors
        seq = features.view(features.size(0), -1, 4)  # each patch as a 4‑dim vector

        # 3. Apply self‑attention (classical implementation)
        attn_out = self.attention(
            self.rotation.detach().cpu().numpy(),
            self.entangle.detach().cpu().numpy(),
            seq.detach().cpu().numpy(),
        )
        attn_tensor = torch.tensor(attn_out, dtype=x.dtype, device=x.device)

        # 4. LSTM sequence modelling
        lstm_out = self.lstm(attn_tensor)   # shape: [B, seq_len, lstm_hidden]

        # 5. Classifier (aggregating over the sequence)
        logits = self.classifier(lstm_out.mean(dim=1))
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
