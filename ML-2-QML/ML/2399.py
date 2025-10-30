"""Hybrid convolution and attention module that can operate in classical mode.

The class mimics the interface of the original Conv filter but augments it with
a self‑attention block.  It is fully PyTorch‑based and can be used as a
drop‑in replacement in existing pipelines.

The design follows a *combination* scaling paradigm: the convolutional
filter is first applied to the input image, its activations are then fed into
a self‑attention mechanism that re‑weights the features.  The attention
weights are computed from two sets of rotation and entangle parameters
passed by the caller, mirroring the signature of the quantum counterpart.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class HybridConvAttention(nn.Module):
    """
    Classical hybrid of a convolutional filter and a self‑attention block.

    Parameters
    ----------
    conv_kernel_size : int
        Size of the square convolution kernel.
    conv_threshold : float
        Threshold used in the sigmoid activation after the convolution.
    attention_dim : int
        Dimensionality of the attention feature space.
    """

    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        attention_dim: int = 4,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=conv_kernel_size, bias=True
        )
        self.threshold = conv_threshold
        self.attention_dim = attention_dim

        # The attention module is implemented with a single‑head
        # multi‑head attention to keep the interface lightweight.
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim, num_heads=1, batch_first=True
        )

    def forward(
        self,
        data: torch.Tensor,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the hybrid module.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape ``(batch, 1, H, W)``.
        rotation_params : np.ndarray, optional
            Parameters used to compute the query vector.
        entangle_params : np.ndarray, optional
            Parameters used to compute the key vector.

        Returns
        -------
        torch.Tensor
            Output of the attention block, shape ``(batch, attention_dim)``.
        """
        # Convolution + sigmoid activation
        conv_out = self.conv(data)
        activated = torch.sigmoid(conv_out - self.threshold)

        # Flatten the feature map and project to the attention space
        batch_size = activated.size(0)
        flat = activated.view(batch_size, -1)

        # Default parameters if not supplied
        if rotation_params is None:
            rotation_params = np.random.randn(self.attention_dim, flat.size(1))
        if entangle_params is None:
            entangle_params = np.random.randn(self.attention_dim, flat.size(1))

        query = torch.from_numpy(flat @ rotation_params.T).float()
        key = torch.from_numpy(flat @ entangle_params.T).float()
        value = flat

        # Compute attention scores
        scores = torch.softmax(query @ key.transpose(0, 1) / np.sqrt(self.attention_dim), dim=-1)
        out = scores @ value

        return out

    def run(
        self,
        data: np.ndarray,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Convenience wrapper that accepts NumPy input and returns NumPy output.

        Parameters
        ----------
        data : np.ndarray
            Input image of shape ``(H, W)``.
        rotation_params : np.ndarray, optional
            Rotation parameters for the attention query.
        entangle_params : np.ndarray, optional
            Entangle parameters for the attention key.

        Returns
        -------
        np.ndarray
            Output of the attention block.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out = self.forward(tensor, rotation_params, entangle_params)
        return out.squeeze(0).numpy()


__all__ = ["HybridConvAttention"]
