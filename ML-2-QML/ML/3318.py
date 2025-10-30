import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2x2 convolution that emulates a quantum-inspired filter."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class SamplerModule(nn.Module):
    """Classical sampler network mirroring the quantum SamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 6),
            nn.Tanh(),
            nn.Linear(6, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class QuanvolutionHybrid(nn.Module):
    """Hybrid classical model that combines a quanvolution filter with a sampler head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.sampler = SamplerModule()
        self.classifier = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        # Use sampler on a reduced representation (first two feature channels)
        sampler_out = self.sampler(features[:, :2])
        gamma = sampler_out[:, 0].unsqueeze(-1)  # scalar gating factor
        logits = self.classifier(features)
        gated_logits = logits * gamma
        return F.log_softmax(gated_logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
