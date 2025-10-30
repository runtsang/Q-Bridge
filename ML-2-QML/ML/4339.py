"""
Classical hybrid fraud detection model that stitches together:

- a quantum‑inspired linear preprocessing layer (FraudLayer)
- an optional quanvolution feature extractor
- a stack of transformer blocks for relational reasoning
- a final classifier head
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """
    Parameters describing a fully‑connected layer inspired by a photonic circuit.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class FraudLayer(nn.Module):
    """
    Quantum‑inspired linear layer that mimics the parameter mapping of a
    photonic circuit.  The first layer is un‑clipped to preserve the raw
    parameter space; subsequent layers are clipped to keep the network
    stable.
    """
    def __init__(self, params: FraudLayerParameters, clip: bool = False) -> None:
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor(params.displacement_r, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        return out * self.scale + self.shift


class QuanvolutionFilter(nn.Module):
    """
    Classical convolutional filter that mimics the structure of the
    quantum quanvolution example but implemented with a 2×2 kernel.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class TransformerBlock(nn.Module):
    """
    Standard transformer block composed of multi‑head self‑attention and
    a two‑layer feed‑forward network.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class FraudDetectionHybrid(nn.Module):
    """
    Classical hybrid fraud detection model that integrates:

    * a quantum‑inspired preprocessing layer (FraudLayer)
    * an optional quanvolution feature extractor
    * a transformer stack for relational reasoning
    * a final classifier head
    """
    def __init__(
        self,
        params: FraudLayerParameters,
        use_quanvolution: bool = False,
        num_features: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_blocks: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.preprocess = FraudLayer(params, clip=False)
        self.use_quanvolution = use_quanvolution

        if use_quanvolution:
            self.quanvolution = QuanvolutionFilter()
            # 4 channels per patch, 14×14 patches for 28×28 input
            self.embed_dim = 4 * 14 * 14
        else:
            self.embed_dim = embed_dim

        self.embedding = nn.Linear(num_features, self.embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlock(self.embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(self.embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2) for tabular data or
            (batch, 1, 28, 28) when using quanvolution.
        """
        x = self.preprocess(x)
        if self.use_quanvolution:
            x = self.quanvolution(x)
        x = self.embedding(x)
        x = self.transformers(x)
        x = x.mean(dim=1)
        return self.classifier(x)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
