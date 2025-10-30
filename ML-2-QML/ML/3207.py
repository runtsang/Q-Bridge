import torch
from torch import nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolutional filter with stride 2, inspired by the original quanvolution example."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Flatten the spatial dimension after convolution
        return self.conv(x).view(x.size(0), -1)


class HybridEstimator(nn.Module):
    """
    Hybrid estimator that first applies a quanvolution filter to extract spatial features
    and then passes them through a compact fully‑connected regressor.
    """
    def __init__(self, in_channels: int = 1, num_outputs: int = 1) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, out_channels=4)
        # For 28×28 MNIST images, 2×2 stride‑2 conv produces 14×14 patches, each with 4 features.
        feature_dim = 4 * 14 * 14
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.Tanh(),
            nn.Linear(32, num_outputs)
        )
        self.num_outputs = num_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        out = self.regressor(features)
        if self.num_outputs == 1:
            return out.squeeze(-1)          # regression output
        return F.log_softmax(out, dim=-1)   # classification output

__all__ = ["HybridEstimator", "QuanvolutionFilter"]
