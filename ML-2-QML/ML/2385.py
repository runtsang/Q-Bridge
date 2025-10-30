"""Hybrid Classical Self‑Attention with Convolutional Pre‑processing.

The module exposes a `SelfAttentionConv()` factory that returns an
instance with a `run` method accepting input data, rotation and
entanglement parameters.  Internally a lightweight PyTorch
convolution filter is applied before the standard scaled‑dot‑product
attention.  The design mirrors the original SelfAttention interface
while adding a convolutional feature extractor, enabling richer
local pattern capture before global weighting.
"""

import numpy as np
import torch
from torch import nn

class ConvFilter(nn.Module):
    """Simple 2‑D convolution filter used as a drop‑in replacement for a
    quantum quanvolution layer.  The filter is applied to each channel
    of the input and its mean activation is returned as a scalar
    feature that is concatenated to the attention input.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data shape: (batch, H, W) or (H, W)
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2,3])  # (batch,1)

class SelfAttentionConv:
    """Hybrid classical self‑attention module with convolutional
    preprocessing.  The output is a weighted sum of the input values
    where the weights are computed from query‑key dot products.
    """
    def __init__(self, embed_dim: int = 4, kernel_size: int = 2):
        self.embed_dim = embed_dim
        self.conv = ConvFilter(kernel_size=kernel_size)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (embed_dim, 3) – rotation angles for each query/key
            dimension.  They are reshaped to (embed_dim, 3) and then
            applied to the input.
        entangle_params : np.ndarray
            Shape (embed_dim-1,) – parameters for the entangling
            gates in the quantum analogue.  They are used here as a
            bias term added to the key matrix.
        inputs : np.ndarray
            Input data of shape (seq_len, embed_dim).  The first
            dimension is treated as a sequence of tokens.
        Returns
        -------
        np.ndarray
            The attended output of shape (seq_len, embed_dim).
        """
        # Convolutional feature extraction
        conv_feat = self.conv.forward(torch.as_tensor(inputs, dtype=torch.float32))
        conv_feat = conv_feat.squeeze(-1)  # (seq_len,)

        # Classical attention computation
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        # Add convolutional bias to key
        key += conv_feat.unsqueeze(1)

        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        return (scores @ value).numpy()

def SelfAttentionConv() -> SelfAttentionConv:
    """Factory that returns a hybrid classical self‑attention instance."""
    return SelfAttentionConv(embed_dim=4, kernel_size=2)

__all__ = ["SelfAttentionConv"]
