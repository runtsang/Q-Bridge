import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomFeatureKernel(nn.Module):
    """Fixed random projection of input features, mimicking a quantum kernel."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        weight = torch.randn(input_dim, output_dim)
        self.register_buffer('weight', weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)

class QuanvolutionHybridClassifier(nn.Module):
    """Classical baseline that combines 2‑D convolution, a random kernel,
    and a linear head. Inspired by the classical and quantum quanvolution
    filters and the QCNN linear head."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 kernel_dim: int = 256):
        super().__init__()
        # 2×2 convolution with stride 2 reduces MNIST 28×28 to 14×14
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2)
        # Flatten then random projection
        self.kernel = RandomFeatureKernel(input_dim=4 * 14 * 14, output_dim=kernel_dim)
        # Final linear classifier
        self.head = nn.Linear(kernel_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.kernel(x)
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["RandomFeatureKernel", "QuanvolutionHybridClassifier"]
