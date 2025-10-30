"""HybridEstimatorQNN: a fully classical hybrid network combining quanvolution, self‑attention and a fully connected head.

The architecture mirrors the quantum‑classical split in the original EstimatorQNN example while
leveraging the SelfAttention, FCL and Quanvolution primitives from the seed repository.
The network is fully differentiable and can be trained with standard PyTorch optimisers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionFilter(nn.Module):
    """Two‑pixel patch encoder that mimics the original quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class SelfAttention(nn.Module):
    """Classical self‑attention block with learnable rotation and entanglement matrices."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = inputs @ self.rotation
        key   = inputs @ self.entangle
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

class FCL(nn.Module):
    """A single linear layer wrapped to expose a run method."""
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def run(self, thetas: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(thetas)).mean(dim=0)

class HybridEstimatorQNN(nn.Module):
    """End‑to‑end classical estimator that chains quanvolution → self‑attention → FCL → output."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.attn    = SelfAttention(embed_dim=4)
        self.fcl     = FCL(in_features=4)
        self.output  = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        features = self.qfilter(x)          # (batch, 4*14*14)
        # Reduce to a fixed‑size embedding for attention
        embed = features.view(features.size(0), -1)[:, :4]
        attn_out = self.attn(embed)         # (batch, 4)
        fcl_out = self.fcl.run(attn_out)    # (batch,)
        logits = self.output(fcl_out.unsqueeze(-1))
        return logits.squeeze(-1)

__all__ = ["HybridEstimatorQNN"]
