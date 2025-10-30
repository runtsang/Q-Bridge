import torch
from torch import nn
import numpy as np

class HybridKernelModel(nn.Module):
    """Hybrid RBF kernel with an optional QCNN‑style feature extractor.

    The kernel is ``exp(-gamma * ||x-y||^2)`` where the inputs can optionally
    be passed through a small convolutional network that mirrors the
    classical QCNN helper.  The class is differentiable and can be used
    in end‑to‑end training pipelines.

    Parameters
    ----------
    gamma : float, default 1.0
        Width of the RBF kernel.
    use_qcnn : bool, default False
        If True, data are processed through a QCNN‑style CNN before the
        kernel computation.
    qcnn_depth : int, default 3
        Number of convolution‑pooling stages in the QCNN feature extractor.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        use_qcnn: bool = False,
        qcnn_depth: int = 3,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_qcnn = use_qcnn
        if use_qcnn:
            self.qcnn = self._build_qcnn(qcnn_depth)

    def _build_qcnn(self, depth: int) -> nn.Module:
        """Build a small QCNN‑style convolutional network."""
        layers = []
        in_dim = 8
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, 16), nn.Tanh(),
                            nn.Linear(16, 8), nn.Tanh(),
                            nn.Linear(8, 4), nn.Tanh()])
            in_dim = 4
        layers.append(nn.Linear(in_dim, 4))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value between *x* and *y*."""
        if self.use_qcnn:
            x = self.qcnn(x)
            y = self.qcnn(y)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the Gram matrix for two batches of samples."""
        a = a.view(a.size(0), -1)
        b = b.view(b.size(0), -1)
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))

    def set_gamma(self, gamma: float) -> None:
        """Update the RBF width."""
        self.gamma = gamma

__all__ = ["HybridKernelModel"]
