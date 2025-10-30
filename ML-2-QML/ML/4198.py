import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFKernel(nn.Module):
    """Classical RBF kernel with optional trainable gamma."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class QuanvolutionFilter(nn.Module):
    """Classical quanvolution filter using 2x2 patches."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class HybridQuantumNAT(nn.Module):
    """Hybrid classical model combining a CNN backbone, quanvolution filter,
    and RBF kernel for similarity augmentation."""
    def __init__(self, num_classes: int = 10, kernel_gamma: float = 1.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.quanv = QuanvolutionFilter()
        self.kernel = RBFKernel(gamma=kernel_gamma)
        self.fc = nn.Linear(4 * 14 * 14 + 1, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor, prototype: torch.Tensor | None = None) -> torch.Tensor:
        features = self.backbone(x)
        flattened = features.view(x.size(0), -1)
        quanv_features = self.quanv(x)
        if prototype is not None:
            sim = self.kernel(flattened, prototype)  # shape (bsz, 1)
            combined = torch.cat([quanv_features, sim], dim=1)
        else:
            combined = quanv_features
        logits = self.fc(combined)
        return self.norm(logits)

__all__ = ["HybridQuantumNAT"]
