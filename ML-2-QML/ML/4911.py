import torch
from torch import nn
import torch.nn.functional as F
from.QuantumKernelMethod import Kernel as ClassicalKernel
from.QuantumNAT import QFCModel as ClassicalQFCModel

class ConvHybrid(nn.Module):
    """
    Classical convolutional module that optionally evaluates a
    radial‑basis kernel between the extracted features and a set of
    reference vectors.  The architecture is inspired by Conv.py,
    QuantumNAT.py and QuantumKernelMethod.py.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 8,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        use_kernel: bool = True,
        gamma: float = 1.0,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.pool = nn.MaxPool2d(2)
        # Fully‑connected head (4 outputs).  Dimension is inferred lazily.
        self._fc = None
        self.use_kernel = use_kernel
        self.kernel = ClassicalKernel(gamma) if use_kernel else None

    def _ensure_fc(self, x: torch.Tensor) -> None:
        if self._fc is None:
            with torch.no_grad():
                feat = self.pool(F.relu(self.conv(x)))
                dim = feat.view(x.size(0), -1).size(1)
            self._fc = nn.Linear(dim, 4)
            self.add_module('_fc', self._fc)

    def forward(self, x: torch.Tensor, refs: torch.Tensor | None = None):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
            refs: Optional tensor of reference vectors for kernel evaluation.
        Returns:
            feat: Tensor (B, 4)
            kernel_mat: Optional kernel matrix (B, N) if refs provided.
        """
        self._ensure_fc(x)
        feat = self.pool(F.relu(self.conv(x)))
        flat = feat.view(x.size(0), -1)
        out = self._fc(flat)
        if self.use_kernel and refs is not None:
            kernel_mat = torch.stack([self.kernel(out[i], refs) for i in range(out.size(0))])
            return out, kernel_mat
        return out

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix between two batches of feature vectors.
        """
        return torch.stack([self.kernel(a[i], b) for i in range(a.size(0))])

__all__ = ["ConvHybrid"]
