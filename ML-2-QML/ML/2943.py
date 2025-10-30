"""Hybrid classical network combining a quanvolution filter and an MLP sampler."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch convolution followed by flattening."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class SamplerModule(nn.Module):
    """Simple MLP that emulates a quantum sampler."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


class QuanvolutionSamplerNet(nn.Module):
    """
    Classical hybrid network:
        quanvolution filter -> linear -> sampler -> linear -> log‑softmax
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.reduce = nn.Linear(4 * 14 * 14, 2)
        self.sampler = SamplerModule(input_dim=2, output_dim=2)
        self.head = nn.Linear(2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.filter(x)
        x = self.reduce(x)
        x = self.sampler(x)
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionSamplerNet"]
