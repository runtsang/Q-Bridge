import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Classical convolutional filter inspired by the quanvolution example."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class ClassicalFullyConnectedQuantumLayer(nn.Module):
    """Classical approximation of a quantum fully‑connected layer."""
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))


class QuanvolutionHybrid(nn.Module):
    """Hybrid neural network that combines a classical quanvolution filter
    with a classical fully‑connected quantum layer approximation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fc_layer = ClassicalFullyConnectedQuantumLayer(4 * 14 * 14)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        features = self.fc_layer(features)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
