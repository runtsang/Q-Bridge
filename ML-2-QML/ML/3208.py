import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch convolution followed by flattening."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class EstimatorQNN(nn.Module):
    """Hybrid estimator that optionally uses a quanvolution filter before a fully‑connected head."""
    def __init__(self,
                 input_dim: int = 2,
                 hidden_sizes: list[int] | None = None,
                 output_dim: int = 1,
                 use_quanvolution: bool = False,
                 in_channels: int = 1,
                 num_classes: int = 10):
        super().__init__()
        self.use_quanvolution = use_quanvolution

        if use_quanvolution:
            self.feature_extractor = QuanvolutionFilter(in_channels=in_channels)
            feature_dim = 4 * 14 * 14  # assuming 28×28 input
            self.head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, output_dim)
            )
        else:
            hidden_sizes = hidden_sizes or [8, 4]
            layers = []
            prev = input_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.Tanh())
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            features = self.feature_extractor(x)
            return self.head(features)
        else:
            return self.net(x)

__all__ = ["QuanvolutionFilter", "EstimatorQNN"]
