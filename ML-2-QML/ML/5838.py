import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

# ------------------------------------------------------------------
# Classical RBF kernel with optional weighting of a quantum kernel
# ------------------------------------------------------------------
class RBFKernel(nn.Module):
    """Classical radial basis function kernel with gamma hyper‑parameter."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute exp(-gamma * ||x - y||^2) for 1‑D tensors."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor]) -> np.ndarray:
        """Return Gram matrix between two datasets."""
        return np.array([[self.forward(torch.tensor(x), torch.tensor(y)).item() for y in Y] for x in X])

# ------------------------------------------------------------------
# Classical quanvolution filter and classifier
# ------------------------------------------------------------------
class QuanvolutionFilter(nn.Module):
    """2‑D convolutional filter mimicking a quanvolution layer."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Classical classifier built on top of the QuanvolutionFilter."""
    def __init__(self, num_classes: int = 10, in_features: int = 4 * 14 * 14) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# ------------------------------------------------------------------
# Unified kernel + quanvolution architecture
# ------------------------------------------------------------------
class UnifiedKernelQuanvolution(nn.Module):
    """
    Combines a classical RBF kernel, an optional quantum kernel, and a quanvolution filter
    into a single module. The kernel can be used for kernel‑based learning, while the
    quanvolution filter provides convolutional feature extraction. The classifier
    head can be chosen to operate on either or both representations.
    """
    def __init__(self,
                 rbf_gamma: float = 1.0,
                 use_quantum_kernel: bool = False,
                 num_classes: int = 10,
                 in_features: int = 4 * 14 * 14) -> None:
        super().__init__()
        self.rbf_kernel = RBFKernel(rbf_gamma)
        self.use_quantum_kernel = use_quantum_kernel
        # Placeholder for a quantum kernel callable; user can inject later
        self.quantum_kernel = None  # type: ignore
        self.classifier = QuanvolutionClassifier(num_classes=num_classes, in_features=in_features)

    def set_quantum_kernel(self, kernel_func) -> None:
        """Inject a quantum kernel callable that accepts two tensors and returns a scalar."""
        self.quantum_kernel = kernel_func
        self.use_quantum_kernel = True

    def combined_kernel_matrix(self,
                               X: Sequence[torch.Tensor],
                               Y: Sequence[torch.Tensor],
                               weight_rbf: float = 0.5,
                               weight_q: float = 0.5) -> np.ndarray:
        """
        Compute a weighted sum of classical RBF and quantum kernels.
        If quantum kernel is not set, returns only the classical part.
        """
        rbf_mat = self.rbf_kernel.matrix(X, Y)
        if self.use_quantum_kernel and self.quantum_kernel is not None:
            q_mat = np.array([[self.quantum_kernel(x, y).item() for y in Y] for x in X])
            return weight_rbf * rbf_mat + weight_q * q_mat
        return rbf_mat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the quanvolution classifier on input images."""
        return self.classifier(x)

__all__ = ["RBFKernel", "QuanvolutionFilter", "QuanvolutionClassifier", "UnifiedKernelQuanvolution"]
