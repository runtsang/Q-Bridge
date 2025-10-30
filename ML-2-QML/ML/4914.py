import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KernalAnsatz(nn.Module):
    """Classical radial‑basis‑function kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz` that normalises inputs."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


class QuanvolutionFilter(nn.Module):
    """Classical 2‑D filter that mimics the original quanvolution block."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class HybridFCL(nn.Module):
    """
    Hybrid fully‑connected layer that chains a quanvolution filter,
    a kernel feature map, and a dense head.
    """
    def __init__(self,
                 in_features: int = 1,
                 kernel_gamma: float = 1.0,
                 n_support: int = 10) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        self.kernel = Kernel(kernel_gamma)
        # Learnable support vectors for the kernel mapping
        self.support = nn.Parameter(torch.randn(n_support, in_features))
        self.fc = nn.Linear(n_support, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quanvolution feature extraction (images)
        if x.dim() == 4:  # batch of 2‑D images
            x = self.quanvolution(x)
        # 2. Kernel feature map
        features = torch.stack([self.kernel(x, sv) for sv in self.support], dim=1)
        # 3. Linear head
        return self.fc(features)

    def run(self, thetas: torch.Tensor) -> np.ndarray:
        """
        Legacy ``run`` interface: accepts a 1‑D array of angles and
        returns the mean activation after a tanh non‑linearity.
        """
        values = thetas.view(-1, 1).float()
        activations = torch.tanh(self.fc(values))
        return activations.mean(dim=0).detach().cpu().numpy()


def FCL() -> HybridFCL:
    """Factory that matches the original API."""
    return HybridFCL()


__all__ = ["KernalAnsatz", "Kernel", "QuanvolutionFilter",
           "HybridFCL", "FCL"]
