import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQuanvolutionFilter(nn.Module):
    """Standard 2×2 convolution followed by flattening."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class ClassicalRegressionHead(nn.Module):
    """Fully‑connected head inspired by the classical regression seed."""
    def __init__(self, input_dim: int,
                 hidden_dims: list[int] | tuple[int,...] = (32, 16)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class QuanvolutionRegression(nn.Module):
    """Hybrid classical model: quanvolution filter + regression head."""
    def __init__(self, num_features: int = 4 * 14 * 14):
        super().__init__()
        self.feature_extractor = ClassicalQuanvolutionFilter()
        self.regressor = ClassicalRegressionHead(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        return self.regressor(feats)

__all__ = ["QuanvolutionRegression"]
