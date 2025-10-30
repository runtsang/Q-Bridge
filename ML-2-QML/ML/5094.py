import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RBFKernel(nn.Module):
    """Radial basis function kernel used as a feature map."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class KernelLayer(nn.Module):
    """Maps high‑dimensional features to a lower‑dimensional kernel space."""
    def __init__(self, num_prototypes: int, feature_dim: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, feature_dim]
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)  # [batch, prototypes, feature_dim]
        dist2 = torch.sum(diff ** 2, dim=-1)  # [batch, prototypes]
        return torch.exp(-self.gamma * dist2)

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution that mimics the quantum filter."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)              # [B, 4, 14, 14]
        return features.view(x.size(0), -1)  # [B, 784]

class QCNNHead(nn.Module):
    """Fully‑connected stack inspired by the QCNN helper."""
    def __init__(self, in_features: int, depth: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        layers = []
        in_dim = in_features
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, in_dim), nn.ReLU()])
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QuanvolutionHybrid(nn.Module):
    """
    Combines a classical quanvolution filter, a kernel feature map,
    and a QCNN‑style classifier head.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        kernel_prototypes: int = 64,
        gamma: float = 1.0,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        # After the filter we obtain 4×14×14 = 784 features
        self.kernel_layer = KernelLayer(kernel_prototypes, 784, gamma=gamma)
        self.head = QCNNHead(kernel_prototypes, depth=depth, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: [B, 1, 28, 28]
        feat = self.filter(x)               # [B, 784]
        ker_feat = self.kernel_layer(feat)  # [B, prototypes]
        logits = self.head(ker_feat)        # [B, num_classes]
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
