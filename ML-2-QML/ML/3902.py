import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQuanvolutionFilter(nn.Module):
    """Classical convolutional filter that extracts 4‑channel features from 28×28 images."""
    def __init__(self) -> None:
        super().__init__()
        # 3×3 conv with stride 2 keeps spatial resolution 14×14
        self.conv = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class ClassicalSamplerQNN(nn.Module):
    """Simple neural sampler that maps 2‑dim inputs to a 2‑dim probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 6),
            nn.ReLU(),
            nn.Linear(6, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridQuanvolutionSampler(nn.Module):
    """
    Hybrid model that concatenates outputs from a classical quanvolution filter
    and a classical sampler, then classifies with a linear head.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter()
        self.sampler = ClassicalSamplerQNN()
        self.linear = nn.Linear(4 * 14 * 14 + 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        # Use the first two features as input to the sampler
        sampler_input = features[:, :2]
        probs = self.sampler(sampler_input)
        combined = torch.cat((features, probs), dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionSampler"]
