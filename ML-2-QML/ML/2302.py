"""Hybrid classical kernel combining quanvolution feature extraction and RBF kernel."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution filter inspired by the quanvolution example."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

class KernalAnsatz(nn.Module):
    """Placeholder maintaining compatibility with the quantum interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

class QuantumKernelQuanvolution(nn.Module):
    """Hybrid classical kernel: quanvolution feature extraction + RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        self.kernel = Kernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return Gram matrix between two batches of images."""
        fx = self.quanvolution(x)  # shape (n, d)
        fy = self.quanvolution(y)  # shape (m, d)
        # pairwise squared distances
        diff = fx.unsqueeze(1) - fy.unsqueeze(0)  # (n, m, d)
        sq = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.kernel.ansatz.gamma * sq).squeeze()

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Convenience wrapper returning a NumPy array."""
        return self.forward(a, b).detach().cpu().numpy()

__all__ = ["QuanvolutionFilter", "Kernel", "QuantumKernelQuanvolution"]
