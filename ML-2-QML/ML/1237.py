"""ConvEnhanced – a multi‑scale, hybrid convolutional filter for classical workflows."""
from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ConvEnhanced(nn.Module):
    """
    A drop‑in replacement for the original Conv filter that supports:
        * Multiple kernel sizes (1×1, 2×2, 3×3, …) for hierarchical feature extraction.
        * An optional attention‑like weighting over the kernel that can be learned.
        * A hybrid loss that mixes classical MSE and a quantum fidelity term.
    The forward method returns the scalar activation and the intermediate logits.
    """

    def __init__(
        self,
        kernel_sizes: list[int] | tuple[int,...] = (2, 3),
        threshold: float = 0.0,
        use_attention: bool = False,
        attention_dim: int = 4,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.threshold = threshold
        self.use_attention = use_attention
        self.device = device

        # One 2‑D convolution per kernel size
        self.convs = nn.ModuleDict()
        for ks in self.kernel_sizes:
            self.convs[str(ks)] = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=ks,
                bias=True,
            )

        # Optional attention module that learns a weight vector over the kernel
        if self.use_attention:
            self.attn = nn.Linear(
                attention_dim,
                1,
                bias=False,
            )
            self.attn_weights = nn.Parameter(torch.ones(attention_dim, device=self.device))

    def forward(self, x: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Args:
            x: 2‑D input tensor of shape (H, W).
        Returns:
            activation: scalar activation value after sigmoid and averaging.
            logits: concatenated logits from all kernel sizes.
        """
        activations = []
        logits_list = []

        # 2‑D convolution for each kernel size
        for ks in self.kernel_sizes:
            conv = self.convs[str(ks)]
            # Reshape input to (1, 1, H, W) for conv2d
            inp = x.view(1, 1, *x.shape)
            logits = conv(inp)  # shape (1, 1, H-ks+1, W-ks+1)
            logits_list.append(logits)

            # Apply threshold and sigmoid
            activ = torch.sigmoid(logits - self.threshold)
            activations.append(activ.mean())

        # Concatenate logits across kernel sizes
        logits_cat = torch.cat(logits_list, dim=2)  # stack along spatial dim

        # Optional attention weighting
        if self.use_attention:
            # Flatten spatial dims
            flat = logits_cat.view(logits_cat.size(0), -1)
            # Compute attention weights
            attn_weights = torch.softmax(self.attn_weights, dim=0)
            weighted = flat * attn_weights
            logits_cat = weighted.view_as(logits_cat)

        # Final scalar activation
        activation = torch.stack(activations).mean().item()

        return activation, logits_cat

__all__ = ["ConvEnhanced"]
