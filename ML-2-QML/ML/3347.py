"""Hybrid self‑attention module combining classical convolutional patching
and a parameterized attention mechanism.

The public API mimics the original SelfAttention helper: a function
`SelfAttention()` returns an instance of :class:`HybridSelfAttention`.
The class accepts an ``embed_dim`` (default 4) and a ``patch_size`` for
image‑like inputs.  Inputs are expected as a NumPy array of shape
``(batch, channels, height, width)`` or flattened patches of shape
``(batch, features)``.  The ``run`` method applies a classical
self‑attention block on the extracted patches, using the supplied
rotation and entanglement parameters to weight the query/key/value
projections.  The implementation uses PyTorch tensors internally for
speed and numerical stability."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def SelfAttention():
    """Factory returning a :class:`HybridSelfAttention` instance."""
    return HybridSelfAttention(embed_dim=4, patch_size=2)


class HybridSelfAttention:
    """Hybrid classical self‑attention with patch extraction."""

    def __init__(self, embed_dim: int = 4, patch_size: int = 2):
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def _extract_patches(self, inputs: np.ndarray) -> torch.Tensor:
        """Convert image‑like inputs into a 2‑D patch tensor.

        Parameters
        ----------
        inputs : np.ndarray
            Shape ``(batch, channels, height, width)`` or
            ``(batch, features)``.  If 4‑D, patches are extracted with
            a stride equal to ``patch_size``.
        Returns
        -------
        torch.Tensor
            Shape ``(batch, num_patches, patch_dim)`` where
            ``patch_dim = channels * patch_size**2``.
        """
        arr = torch.as_tensor(inputs, dtype=torch.float32)
        if arr.ndim == 4:
            # image → patches
            b, c, h, w = arr.shape
            kernel = self.patch_size
            patches = F.unfold(arr, kernel_size=kernel, stride=kernel)
            num_patches = patches.shape[-1]
            patches = patches.transpose(1, 2).reshape(b, num_patches, -1)
        else:
            # already flattened
            patches = arr.unsqueeze(1)
        return patches

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Apply the hybrid attention block.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(embed_dim * patch_dim,)`` – linear projection
            parameters for the query and key.
        entangle_params : np.ndarray
            Shape ``(embed_dim * patch_dim,)`` – linear projection
            parameters for the value.
        inputs : np.ndarray
            Image or flattened patch data.

        Returns
        -------
        np.ndarray
            Output of shape ``(batch, embed_dim)`` – the attended
            representation of each input sample.
        """
        patches = self._extract_patches(inputs)  # (b, n, d)
        b, n, d = patches.shape
        # Linear projections
        q = torch.matmul(patches, torch.as_tensor(
            rotation_params.reshape(d, self.embed_dim), dtype=torch.float32))
        k = torch.matmul(patches, torch.as_tensor(
            entangle_params.reshape(d, self.embed_dim), dtype=torch.float32))
        v = patches  # use raw patches as values
        # Attention scores
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        # Weighted sum
        out = torch.matmul(scores, v)  # (b, n, d)
        # Aggregate over patches (mean)
        out = out.mean(dim=1)  # (b, d)
        return out.numpy()


__all__ = ["SelfAttention", "HybridSelfAttention"]
