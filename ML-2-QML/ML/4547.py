"""Hybrid classical LSTM model with optional convolutional preprocessing and fraud‑detection layers.

This module merges concepts from:
- Classical LSTM implementation
- Convolutional filtering (mimicking a quantum quanvolution)
- Fraud detection inspired parameterised linear layers
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Simple 2‑D convolutional filter that mimics a quantum quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor) -> float:
        """Apply the filter and return a scalar activation."""
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class FraudLayer(nn.Module):
    """Parameterised linear layer with optional clipping, activation, scaling and shifting."""
    def __init__(self,
                 bs_theta: float,
                 bs_phi: float,
                 phases: tuple[float, float],
                 squeeze_r: tuple[float, float],
                 squeeze_phi: tuple[float, float],
                 displacement_r: tuple[float, float],
                 displacement_phi: tuple[float, float],
                 kerr: tuple[float, float],
                 clip: bool = False) -> None:
        super().__init__()
        weight = torch.tensor([[bs_theta, bs_phi],
                               [squeeze_r[0], squeeze_r[1]]], dtype=torch.float32)
        bias = torch.tensor(phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(displacement_phi, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.activation(self.linear(inputs))
        out = out * self.scale + self.shift
        return out

def build_fraud_detection_program(input_params, layers):
    """Create a sequential model mirroring the photonic fraud‑detection structure."""
    modules = [FraudLayer(**vars(input_params), clip=False)]
    modules += [FraudLayer(**vars(l), clip=True) for l in layers]
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class HybridQLSTM(nn.Module):
    """Drop‑in replacement for a sequence‑tagger that can operate in classical mode
    and optionally use convolutional preprocessing or fraud‑detection style feature
    extraction.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 conv_kernel: int | None = None,
                 fraud_params: tuple | None = None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Classical LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Optional convolutional preprocessing
        self.conv = ConvFilter(kernel_size=conv_kernel) if conv_kernel else None

        # Optional fraud‑detection style feature extractor
        self.fraud = build_fraud_detection_program(*fraud_params) if fraud_params else None

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # sentence shape: (seq_len, batch)
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embedding_dim)

        # Pre‑process each embedding if convolution is enabled
        if self.conv:
            seq_len, batch, dim = embeds.shape
            # assume dim == conv_kernel ** 2
            patch = embeds.view(seq_len, batch, self.conv.kernel_size, self.conv.kernel_size)
            conv_out = torch.zeros(seq_len, batch, 1, device=embeds.device)
            for t in range(seq_len):
                conv_out[t] = self.conv.run(patch[t].squeeze(1))
            embeds = conv_out.squeeze(-1).unsqueeze(-1)  # back to (seq_len, batch, 1)

        lstm_out, _ = self.lstm(embeds)
        # Fraud‑detection feature extraction
        if self.fraud:
            lstm_out = self.fraud(lstm_out)

        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["HybridQLSTM"]
