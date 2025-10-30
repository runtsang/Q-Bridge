"""Hybrid classical module combining self‑attention, convolution and kernel methods.

The design follows the original SelfAttention reference but expands
to include a learnable convolutional filter (ConvFilter) and an
RBF kernel (Kernel).  Each component can be called independently
or combined into a single forward pass.  The implementation
keeps the public API compatible with the original anchor
(`SelfAttention()` returns an instance that exposes a `run`
method) while adding new capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

class ConvFilter(nn.Module):
    """Convolutional filter that emulates a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        """Apply the filter to a 2‑D array and return a scalar."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class Kernel(nn.Module):
    """Radial‑basis function kernel implemented as a PyTorch module."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class SelfAttentionHybrid(nn.Module):
    """Hybrid self‑attention module that integrates attention,
    convolution and kernel operations."""
    def __init__(
        self,
        embed_dim: int = 4,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Auxiliary modules
        self.conv = ConvFilter(kernel_size, conv_threshold)
        self.kernel = Kernel(kernel_gamma)

    def forward(self, inputs: torch.Tensor, rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        """
        Compute a weighted sum of three sub‑modules:
            1. Classical self‑attention
            2. Convolutional filter applied to the raw input
            3. RBF kernel similarity between the input and a
               learned reference vector.

        Parameters
        ----------
        inputs : torch.Tensor
            Batch of input vectors (batch, embed_dim).
        rotation_params, entangle_params : np.ndarray
            Parameters used only for the attention part; they are
            reshaped to match the projection matrices.

        Returns
        -------
        torch.Tensor
            Combined representation of shape (batch, embed_dim).
        """
        # 1. Classical self‑attention
        rot = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        ent = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        query = self.query_proj(inputs @ rot)
        key   = self.key_proj(inputs @ ent)
        value = self.value_proj(inputs)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ value

        # 2. Convolutional filter on the first feature map
        #    (assumes inputs[0] can be reshaped to 2‑D)
        conv_out = torch.tensor(
            self.conv.run(inputs[0].detach().cpu().numpy()),
            dtype=torch.float32,
        ).unsqueeze(0).expand_as(attn_out)

        # 3. Kernel similarity against a learned reference
        #    (here we use the first sample as reference for simplicity)
        ref = inputs[0].unsqueeze(0)
        kernel_out = self.kernel(inputs, ref).expand_as(attn_out)

        # Combine: simple averaging; could be learned weighted sum
        return (attn_out + conv_out + kernel_out) / 3.0

    # Convenience wrappers matching the original API
    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """Legacy interface that accepts numpy arrays."""
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        out = self.forward(inputs_t, rotation_params, entangle_params)
        return out.detach().cpu().numpy()

def SelfAttention() -> SelfAttentionHybrid:
    """Factory that returns a pre‑configured hybrid module."""
    return SelfAttentionHybrid()
