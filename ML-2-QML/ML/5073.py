"""Hybrid classical estimator that mimics a quantum neural network.

The architecture stitches together:
• a lightweight fully‑connected layer (FCL) that acts as a learnable
  pre‑processor,
• a quantum‑inspired convolutional filter (QuanvolutionFilter) that
  extracts spatial kernels from 2×2 patches,
• a self‑attention module that weighs the resulting feature map,
• and a final linear head that produces a scalar regression output.

By replacing the quantum sub‑components with classical approximations
the network can be trained on a CPU while still exposing the same
forward‑pass semantics as the quantum version.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable

# ----------------------------------------------------------------------
# Classical stand‑ins for the quantum primitives
# ----------------------------------------------------------------------
class FCL(nn.Module):
    """Learnable linear mapping that mimics a single‑qubit rotation."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean(dim=0).detach().numpy()


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution that emulates a quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)


class SelfAttentionTorch(nn.Module):
    """Self‑attention block implemented with standard torch ops."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query_lin = nn.Linear(embed_dim, embed_dim)
        self.key_lin   = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        query = self.query_lin(inputs)
        key   = self.key_lin(inputs)
        scores = torch.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs


# ----------------------------------------------------------------------
# Hybrid estimator
# ----------------------------------------------------------------------
class EstimatorQNNHybrid(nn.Module):
    """
    A hybrid estimator that unites classical approximations of the
    quantum components used in the original EstimatorQNN example.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 embed_dim: int = 4,
                 n_classes: int = 1) -> None:
        super().__init__()

        # 1. Pre‑processing (quantum‑inspired FC)
        self.preproc = FCL(n_features=input_dim)

        # 2. Convolutional feature extractor
        self.qfilter = QuanvolutionFilter()

        # 3. Self‑attention mechanism
        self.attention = SelfAttentionTorch(embed_dim=embed_dim)

        # 4. Output head
        # The number of features after the filter is 4 * 14 * 14
        self.output = nn.Linear(4 * 14 * 14, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Pre‑processing expects a 1‑D array of parameters
        prep = torch.tensor(self.preproc.run(x.squeeze().tolist()), dtype=torch.float32)
        # Expand to batch dimension
        prep = prep.unsqueeze(0)

        # Convolution
        conv = self.qfilter(prep.unsqueeze(1))  # add channel dim

        # Attention: reshape to (batch, embed_dim, other)
        embed_dim = self.attention.embed_dim
        # For simplicity, split the feature map into embed_dim chunks
        chunks = conv.chunk(embed_dim, dim=1)
        attn_out = torch.cat([self.attention(chunk) for chunk in chunks], dim=1)

        # Final linear layer
        return self.output(attn_out)


__all__ = ["EstimatorQNNHybrid"]
