"""Hybrid classical self‑attention module that blends convolutional patch extraction
with a standard scaled‑dot‑product attention mechanism.

The implementation mirrors the interface of the original SelfAttention helper
while adding a learnable convolutional front‑end inspired by the Quanvolution
filter.  It accepts ``rotation_params`` and ``entangle_params`` (used only for
compatibility with the quantum variant) and produces an attention‑weighted
feature map that can be fed into downstream classifiers.

The class is fully NumPy/Torch‑based and is deliberately lightweight so it can
be dropped into any deep‑learning pipeline without external quantum
dependencies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSelfAttention:
    """
    Classical self‑attention block with a convolutional patch extractor.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of the query/key/value vectors.
    patch_size : int, default 2
        Size of the square patch to convolve over the input image.
    """

    def __init__(self, embed_dim: int, patch_size: int = 2) -> None:
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        # Convolution that extracts non‑overlapping patches
        self.patch_conv = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute attention‑weighted features.

        Parameters
        ----------
        rotation_params : np.ndarray
            Unused in the classical branch but accepted for API parity.
        entangle_params : np.ndarray
            Unused in the classical branch but accepted for API parity.
        inputs : np.ndarray
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        np.ndarray
            Attention‑weighted feature map of shape (batch, embed_dim).
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Extract patches -> (batch, embed_dim, H', W')
        patches = self.patch_conv(x)

        # Flatten spatial dimensions -> (batch, seq_len, embed_dim)
        seq_len = patches.shape[2] * patches.shape[3]
        patches = patches.view(patches.shape[0], self.embed_dim, seq_len).transpose(1, 2)

        # Compute query, key, value
        q = torch.matmul(patches, torch.as_tensor(rotation_params.T, dtype=torch.float32))
        k = torch.matmul(patches, torch.as_tensor(entangle_params.T, dtype=torch.float32))
        v = patches

        # Scaled dot‑product attention
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)

        # Weighted sum of values
        weighted = torch.matmul(scores, v)
        # Collapse sequence dimension (attention over patches)
        out = weighted.sum(dim=1)

        return out.detach().cpu().numpy()


__all__ = ["HybridSelfAttention"]
