"""Enhanced classical quanvolution with learnable preprocessing and Bayesian head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class PreProcessor(nn.Module):
    """Normalises each 2×2 patch before convolution."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(-1, 1, 2, 2)  # (B*14*14, 1, 2, 2)
        mean = patches.mean(dim=(2, 3), keepdim=True)
        std = patches.std(dim=(2, 3), keepdim=True) + 1e-6
        patches = (patches - mean) / std
        return patches

class QuanvolutionFilter(nn.Module):
    """Learnable convolution over normalised 2×2 patches."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        preproc = PreProcessor()(x)
        B = x.shape[0]
        preproc = preproc.view(B, 1, 28, 28)
        out = self.conv(preproc)  # (B, 4, 14, 14)
        return out.view(B, -1)

class BayesianLinear(nn.Module):
    """Simple Bayesian linear layer with reparameterisation."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.mu_weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.logvar_weight = nn.Parameter(torch.full((in_features, out_features), -3.0))
        self.mu_bias = nn.Parameter(torch.zeros(out_features))
        self.logvar_bias = nn.Parameter(torch.full((out_features,), -3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = Normal(self.mu_weight, torch.exp(0.5 * self.logvar_weight)).rsample()
        bias = Normal(self.mu_bias, torch.exp(0.5 * self.logvar_bias)).rsample()
        return F.linear(x, weight, bias)

class QuanvolutionClassifier(nn.Module):
    """Hybrid network: quanvolution filter → Bayesian head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.bayes_head = BayesianLinear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.bayes_head(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier", "BayesianLinear", "PreProcessor"]
