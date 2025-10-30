import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """Classical hybrid network that emulates a quantum filter using a
    2×2 convolution with optional thresholding and a multi‑layer
    feed‑forward head.  The design is inspired by the original
    quanvolution example and the Conv.py filter, while scaling
    with arbitrary kernel size, output depth, and network depth."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        threshold: float = 0.0,
        depth: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Classical “quantum” filter
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size
        )

        # Multi‑layer head inspired by build_classifier_circuit
        flat_features = out_channels * (28 // kernel_size) ** 2
        layers = []
        in_dim = flat_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolution and optional threshold
        features = self.conv(x)
        if self.threshold!= 0.0:
            features = torch.sigmoid(features - self.threshold)
        flat = features.view(x.size(0), -1)
        logits = self.head(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
