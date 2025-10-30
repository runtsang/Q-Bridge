import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionRegressionModel(nn.Module):
    """
    Classical quanvolution filter followed by an MLP regression head.
    """
    def __init__(self,
                 in_channels: int = 1,
                 conv_out_channels: int = 4,
                 kernel_size: int = 2,
                 stride: int = 2,
                 hidden_dim: int = 64,
                 out_features: int = 1):
        super().__init__()
        self.qfilter = nn.Conv2d(in_channels, conv_out_channels,
                                 kernel_size=kernel_size, stride=stride)
        feature_map_size = 28 // stride  # 14
        self.mlp = nn.Sequential(
            nn.Linear(conv_out_channels * feature_map_size * feature_map_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        flattened = features.view(x.size(0), -1)
        return self.mlp(flattened)

__all__ = ["QuanvolutionRegressionModel"]
