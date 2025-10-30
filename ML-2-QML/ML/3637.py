import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalQuanvolutionFilter(nn.Module):
    """Convolutional filter mimicking the 2×2 patch size of the quantum quanvolution."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x shape: (batch, 1, 28, 28)
        out = self.conv(x)          # (batch, 4, 14, 14)
        return out.view(x.size(0), -1)  # flatten to (batch, 4*14*14)


class HybridEstimatorNN(nn.Module):
    """Classical regression head built on top of the classical quanvolution filter."""
    def __init__(self, n_features: int = 4 * 14 * 14, n_outputs: int = 1) -> None:
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter()
        self.linear = nn.Linear(n_features, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        return self.linear(features)


def EstimatorQNN() -> HybridEstimatorNN:
    """Return an instantiated hybrid estimator."""
    return HybridEstimatorNN()


__all__ = ["EstimatorQNN"]
