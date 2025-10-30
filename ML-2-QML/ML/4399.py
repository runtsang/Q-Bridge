import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block operating on a batch of feature vectors."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(x)
        key   = self.key_proj(x)
        value = self.value_proj(x)
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolution that mimics the structure of the quantum filter."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        return feat.view(x.size(0), -1)

class EstimatorQNNHybrid(nn.Module):
    """Hybrid feed‑forward regressor that optionally uses a quanvolution filter
    and a classical self‑attention module before the final linear head."""
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: list[int] | None = None,
                 use_attention: bool = True,
                 use_quanvolution: bool = False):
        super().__init__()
        hidden_dims = hidden_dims or [8, 4]
        self.use_attention = use_attention
        self.use_quanvolution = use_quanvolution

        if self.use_quanvolution:
            self.qfilter = QuanvolutionFilter()
            in_features = 4 * 14 * 14  # 28x28 image → 14x14 patches → 4 channels
        else:
            in_features = input_dim

        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        if self.use_attention:
            self.attn = ClassicalSelfAttention(embed_dim=hidden_dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            x = self.qfilter(x)

        # First hidden layer produces intermediate features
        feat = self.net[0](x)
        feat = self.net[1](feat)

        if self.use_attention:
            feat = self.attn(feat)

        # Continue with the remaining layers
        for layer in self.net[2:]:
            feat = layer(feat)

        return feat

def EstimatorQNN() -> EstimatorQNNHybrid:
    """Return a hybrid estimator that combines feed‑forward, attention
    and optional quanvolution.  Mirrors the original EstimatorQNN API."""
    return EstimatorQNNHybrid()

__all__ = ["EstimatorQNN", "EstimatorQNNHybrid",
           "ClassicalSelfAttention", "QuanvolutionFilter"]
