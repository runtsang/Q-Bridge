"""Combined classical sampler that chains convolution, auto‑encoding and a small MLP sampler.

The class mirrors the quantum counterpart and can be used as a drop‑in
replacement for the original `SamplerQNN` while providing a richer feature
extraction pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the lightweight modules from the seed repository
from Conv import Conv
from Autoencoder import Autoencoder
from QuantumKernelMethod import Kernel

class SharedClassName(nn.Module):
    """Hybrid classical sampler that chains convolution, auto‑encoding and a small MLP sampler.

    Parameters
    ----------
    conv_params : dict, optional
        Arguments for :class:`Conv`. Default ``{'kernel_size': 2}``.
    autoencoder_params : dict, optional
        Arguments for :class:`Autoencoder`. The ``input_dim`` is inferred from
        ``conv_params['kernel_size']``.
    sampler_params : dict, optional
        Parameters for the final sampler MLP. Default ``{'hidden_dim': 64,
        'output_dim': 2}``.
    """
    def __init__(
        self,
        conv_params: dict | None = None,
        autoencoder_params: dict | None = None,
        sampler_params: dict | None = None,
    ) -> None:
        super().__init__()

        conv_params = conv_params or {}
        self.kernel_size = conv_params.get("kernel_size", 2)
        self.conv = Conv()
        self.conv_layer = self.conv.conv  # underlying nn.Conv2d

        # Autoencoder input dimension is the flattened convolution output
        autoencoder_params = autoencoder_params or {}
        autoencoder_params.setdefault("input_dim", self.kernel_size ** 2)
        self.autoencoder = Autoencoder(**autoencoder_params)

        sampler_params = sampler_params or {}
        hidden_dim = sampler_params.get("hidden_dim", 64)
        output_dim = sampler_params.get("output_dim", 2)

        self.sampler = nn.Sequential(
            nn.Linear(autoencoder_params.get("latent_dim", 32), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1),
        )

        # Optional kernel module for similarity queries
        self.kernel = Kernel(gamma=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape ``(batch, 1, H, W)``. The convolution
            operates on each image patch independently.

        Returns
        -------
        torch.Tensor
            Probabilities of shape ``(batch, output_dim)``.
        """
        # Convolution
        conv_out = self.conv_layer(x)  # shape (batch, 1, k, k)
        conv_out = conv_out.view(x.size(0), -1)  # flatten

        # Auto‑encoding
        latent = self.autoencoder.encode(conv_out)

        # Sampler MLP
        probs = self.sampler(latent)
        return probs

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix between two sets of feature vectors using the
        underlying RBF kernel.

        Parameters
        ----------
        a, b : torch.Tensor
            Feature tensors of shape ``(n, d)`` and ``(m, d)``.

        Returns
        -------
        torch.Tensor
            Gram matrix of shape ``(n, m)``.
        """
        return self.kernel(a, b)

__all__ = ["SharedClassName"]
