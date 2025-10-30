"""Hybrid classical self‑attention with quanvolution and classifier.

The class combines the classical SelfAttention helper, a
Quanvolution‑style convolution, a linear classification head,
and an optional regression head.  Parameter shapes follow the
original SelfAttention interface so that existing pipelines
continue to work unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSelfAttention(nn.Module):
    """Combines classical self‑attention, quanvolution, and a linear head.

    Parameters
    ----------
    embed_dim : int
        Dimension of the attention embedding.  Defaults to 4.
    conv_out_channels : int
        Number of output channels for the quanvolution convolution.
        Defaults to 4.
    num_classes : int
        Number of classes for the classification head.
    regression : bool
        If ``True`` a regression head is added.
    """
    def __init__(self,
                 embed_dim: int = 4,
                 conv_out_channels: int = 4,
                 num_classes: int = 10,
                 regression: bool = False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(1, conv_out_channels, kernel_size=2, stride=2)
        # Self‑attention implemented as a matrix‑product softmax
        self.classifier = nn.Linear(conv_out_channels * 14 * 14, num_classes)
        self.regressor = nn.Linear(conv_out_channels * 14 * 14, 1) if regression else None

    def forward(self,
                x: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).
        rotation_params : np.ndarray
            Parameters for the query projection.  Shape compatible with
            ``(embed_dim, 1)`` or ``(embed_dim,)``.
        entangle_params : np.ndarray
            Parameters for the key projection.  Shape compatible with
            ``(embed_dim, 1)`` or ``(embed_dim,)``.
        Returns
        -------
        torch.Tensor
            Classification logits (and regression output if enabled).
        """
        # Quanvolution‑like convolution
        conv_out = self.conv(x)  # (batch, conv_out_channels, 14, 14)
        flat = conv_out.view(conv_out.size(0), -1)  # (batch, features)

        # Self‑attention on the flattened features
        query = torch.as_tensor(flat @ rotation_params.reshape(self.embed_dim, -1),
                                dtype=torch.float32)
        key = torch.as_tensor(flat @ entangle_params.reshape(self.embed_dim, -1),
                              dtype=torch.float32)
        value = flat
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ value

        # Classification (and optional regression)
        logits = self.classifier(attn_out)
        if self.regressor is not None:
            reg = self.regressor(attn_out)
            return logits, reg
        return logits

__all__ = ["HybridSelfAttention"]
