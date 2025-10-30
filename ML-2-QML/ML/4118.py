"""Hybrid Self‑Attention integrating convolution and LSTM gating (classical implementation)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvFilter(nn.Module):
    """2×2 convolutional feature extractor used as a lightweight attention gate."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape (batch, 1, kernel_size, kernel_size)

        Returns
        -------
        torch.Tensor
            Scalar feature per batch element.
        """
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


class QLSTMGate(nn.Module):
    """LSTM gate that maps a scalar feature to a gate value in (0,1)."""
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, seq_len, input_dim)

        Returns
        -------
        torch.Tensor
            Gate values of shape (batch, seq_len, 1).
        """
        out, _ = self.lstm(x)
        gate = torch.sigmoid(self.linear(out))
        return gate


class SelfAttention(nn.Module):
    """
    Classical self‑attention that first extracts a scalar feature from each token
    via a 2×2 convolution, then uses an LSTM gate to produce dynamic attention
    weights.  The resulting weighted sum is returned as the sequence representation.
    """
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 conv_kernel: int = 2,
                 lstm_hidden: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.conv = ConvFilter(kernel_size=conv_kernel)
        self.lstm_gate = QLSTMGate(hidden_dim=lstm_hidden)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim)

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, embed_dim) – the attended representation.
        """
        batch, seq_len, _ = inputs.shape

        # Compute queries, keys, values
        q = self.query(inputs)  # (batch, seq_len, embed_dim)
        k = self.key(inputs)
        v = self.value(inputs)

        # Raw attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = torch.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # Convolutional feature per token
        conv_features = []
        for i in range(seq_len):
            if self.embed_dim == 4:
                data_i = inputs[:, i, :].view(batch, 1, 2, 2)
                feat = self.conv(data_i)  # (batch,)
            else:
                feat = torch.sigmoid(inputs[:, i, :].sum(dim=-1, keepdim=True)).squeeze(-1)
            conv_features.append(feat)
        conv_features = torch.stack(conv_features, dim=1)  # (batch, seq_len)

        # LSTM gating on convolutional features
        gate = self.lstm_gate(conv_features.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len, 1)

        # Apply gate to attention and renormalise
        gated_attn = attn * gate  # broadcast over last dimension
        gated_attn = gated_attn / gated_attn.sum(dim=-1, keepdim=True)

        # Weighted sum of values
        out = torch.matmul(gated_attn, v)  # (batch, seq_len, embed_dim)
        return out.mean(dim=1)  # aggregate across sequence

__all__ = ["SelfAttention"]
