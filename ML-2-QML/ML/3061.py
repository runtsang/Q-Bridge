"""Hybrid classical network combining quanvolution filter and sampler."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2x2 convolution with stride 2."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class SamplerModule(nn.Module):
    """Classical sampler network mirroring SamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class QuanvolutionClassifier(nn.Module):
    """Classifier head for quanvolution features."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

class QuanvolutionSamplerNet(nn.Module):
    """Hybrid network: quanvolution filter → linear head → quantum-inspired sampler."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)
        self.sampler = SamplerModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        probs = F.log_softmax(logits, dim=-1)
        # Use first two logits as inputs to sampler
        sampler_input = logits[:, :2]
        sampler_output = self.sampler(sampler_input)
        return torch.cat([probs, sampler_output], dim=-1)

__all__ = [
    "QuanvolutionFilter",
    "SamplerModule",
    "QuanvolutionClassifier",
    "QuanvolutionSamplerNet",
]
