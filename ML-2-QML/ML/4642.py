import torch
from torch import nn
import numpy as np

class HybridConv(nn.Module):
    """
    Classical hybrid convolution that emulates the original Conv filter,
    enriches the feature map with an RBF kernel against a fixed basis,
    and feeds the kernel vector through a tiny MLP sampler.

    Parameters
    ----------
    kernel_size : int
        Size of the 2‑D convolution filter (default 2).
    threshold : float
        Sigmoid threshold applied to the conv logits.
    gamma : float
        RBF kernel width.
    basis_size : int
        Number of fixed basis vectors for kernel feature extraction.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 gamma: float = 1.0,
                 basis_size: int = 8,
                 device: str = 'cpu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.gamma = gamma
        self.device = device

        # 1‑channel 1‑output convolution (drop‑in for ConvFilter)
        self.conv = nn.Conv2d(1, 1,
                              kernel_size=kernel_size,
                              bias=True,
                              device=device)

        # Fixed, non‑trainable RBF basis vectors
        self.register_buffer(
            'basis',
            torch.randn(basis_size, kernel_size * kernel_size, device=device)
        )

        # Lightweight MLP sampler that turns kernel features into a scalar
        self.sampler = nn.Sequential(
            nn.Linear(basis_size, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute a single RBF kernel value."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def run(self, data: np.ndarray) -> float:
        """
        Run the hybrid filter on a 2‑D numpy array.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Scalar prediction from the sampler network.
        """
        # Forward pass through convolution
        tensor = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold).view(-1)

        # RBF kernel features against each basis vector
        features = torch.cat([self._rbf(activations, b) for b in self.basis], dim=0)
        features = features.unsqueeze(0)  # batch dim

        # MLP sampler output
        out = self.sampler(features)
        return out.item()

__all__ = ["HybridConv"]
