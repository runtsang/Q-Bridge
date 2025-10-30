import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGen221(nn.Module):
    """
    Classical CNN that emulates a quanvolution layer with a Conv2d followed by a sigmoid threshold.
    Builds on the feature extractor from Quantum‑NAT and ends with a fully‑connected head.
    """
    def __init__(self,
                 in_channels: int = 1,
                 conv_channels: int = 8,
                 kernel_size: int = 3,
                 pool_size: int = 2,
                 filter_kernel: int = 2,
                 threshold: float = 0.0,
                 fc_hidden: int = 64,
                 output_dim: int = 4):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=kernel_size,
                      stride=1, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=kernel_size,
                      stride=1, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
            # Quantum‑inspired filter
            nn.Conv2d(conv_channels * 2, conv_channels * 2,
                      kernel_size=filter_kernel, stride=1, bias=True),
        )
        self.threshold = threshold
        # Compute flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 28, 28)
            out = self.features(dummy)
            self.flat_dim = out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, output_dim)
        )
        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # Apply quantum‑inspired sigmoid threshold
        x = torch.sigmoid(x - self.threshold)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.norm(x)

__all__ = ["ConvGen221"]
