import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSampler(nn.Module):
    """Simple classical sampler generating a 2‑dim probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridQuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution followed by flattening."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class HybridQuanvolutionClassifier(nn.Module):
    """Hybrid classifier that augments convolutional features with a sampler output."""
    def __init__(self) -> None:
        super().__init__()
        self.filter = HybridQuanvolutionFilter()
        self.sampler = HybridSampler()
        self.linear = nn.Linear(4 * 14 * 14 + 2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        # Dummy 2‑dim input for the sampler (e.g., zeros)
        dummy = torch.zeros(x.size(0), 2, device=x.device)
        sampler_features = self.sampler(dummy)
        combined = torch.cat([features, sampler_features], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionFilter", "HybridQuanvolutionClassifier"]
