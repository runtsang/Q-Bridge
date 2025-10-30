import torch
from torch import nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Classical 2‑D convolution simulating the quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, out_channels: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, out_channels, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

class QuanvolutionFilter(nn.Module):
    """Classical approximation of a quanvolution filter."""
    def __init__(self, num_wires: int = 4, out_channels: int = 4) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.conv = nn.Conv2d(1, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class HybridConvNet(nn.Module):
    """
    Classical hybrid network combining a depthwise convolution, a
    quantum‑inspired filter, and a linear head.

    The architecture follows the patterns from the three reference pairs:
    * a base ConvFilter (from Conv.py) to extract local statistics,
    * a QuanvolutionFilter (from Quanvolution.py) to emulate a quantum
      convolution with random two‑qubit layers,
    * a fully‑connected head that can be used for classification or
      regression.

    This class can be dropped into existing pipelines that expect a
    Conv module while providing the expressive power of a quantum
    filter in a purely classical implementation.
    """
    def __init__(
        self,
        in_channels: int = 1,
        conv_out_channels: int = 1,
        quanv_out_channels: int = 4,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=2, out_channels=conv_out_channels)
        self.quanv = QuanvolutionFilter(num_wires=4, out_channels=quanv_out_channels)
        # Compute head dimension using a dummy input
        dummy = torch.zeros(1, in_channels, 28, 28)
        feat = self.forward_features(dummy)
        self.head = nn.Linear(feat.shape[1], num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.quanv(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)
