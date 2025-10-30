import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridSamplerQNN(nn.Module):
    """
    Classical sampler that combines a quanvolution-like convolution,
    an RBF kernel embedding, and a linear regression head.
    """
    def __init__(self, support_vectors: torch.Tensor, gamma: float = 1.0):
        """
        Parameters
        ----------
        support_vectors : torch.Tensor
            Reference vectors used for the RBF kernel. Shape: (n_support, dim).
        gamma : float, optional
            RBF kernel width, by default 1.0.
        """
        super().__init__()
        self.support = support_vectors
        self.gamma = gamma
        # Linear head mapping kernel similarity to 2-class logits
        self.head = nn.Linear(self.support.size(0), 2)

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix between `x` and `y`.

        Parameters
        ----------
        x : torch.Tensor
            Input batch. Shape: (batch, dim).
        y : torch.Tensor
            Support vectors. Shape: (n_support, dim).

        Returns
        -------
        torch.Tensor
            Kernel similarity matrix. Shape: (batch, n_support).
        """
        diff = x.unsqueeze(1) - y.unsqueeze(0)          # (batch, n_support, dim)
        dist_sq = (diff ** 2).sum(-1)                   # (batch, n_support)
        return torch.exp(-self.gamma * dist_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a softmax probability distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input batch. Shape: (batch, dim).

        Returns
        -------
        torch.Tensor
            Probabilities over two classes. Shape: (batch, 2).
        """
        K = self.rbf_kernel(x, self.support)            # (batch, n_support)
        logits = self.head(K)                          # (batch, 2)
        return F.softmax(logits, dim=-1)

__all__ = ["HybridSamplerQNN"]
