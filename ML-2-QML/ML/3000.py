import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 filter with stride 2 that mimics the original quanvolution."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class HybridQuanvolutionNet(nn.Module):
    """
    Purely classical binary classifier that uses a quanvolution filter as a feature extractor
    and a linear head to produce class probabilities.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # 4 filters × 14 × 14 patches = 784 features
        self.head = nn.Linear(4 * 14 * 14, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.head(features)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuanvolutionFilter", "HybridQuanvolutionNet"]
