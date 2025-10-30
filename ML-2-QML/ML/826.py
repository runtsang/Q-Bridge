"""Enhanced classical quanvolution filter and classifier.

The module keeps the 2×2 patch extraction of the original example but replaces the fixed convolution with a learnable 1‑D projection per patch.  
A gated‑recurrent unit (GRU) processes the sequence of patch embeddings, allowing the network to capture spatial dependencies across the image grid before the final linear classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Learnable 2‑D to 1‑D patch encoder.

    * A 2×2 patch is extracted by a stride‑2 convolution.
    * Each patch is projected to a feature vector via a linear layer.
    * The output shape is (batch, num_patches, patch_dim).
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_out_channels: int = 4,
        patch_dim: int = 16,
        kernel_size: int = 2,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, conv_out_channels, kernel_size=kernel_size, stride=stride
        )
        # The number of 2×2 patches in a 28×28 image is 14×14=196
        self.num_patches = 14 * 14
        # Linear projection from conv_out_channels to patch_dim
        self.proj = nn.Linear(conv_out_channels, patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W) where H=W=28.

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape (B, num_patches, patch_dim).
        """
        # Conv produces (B, conv_out_channels, 14, 14)
        patches = self.conv(x)  # shape (B, conv_out_channels, 14, 14)
        # Flatten spatial dimensions
        patches = patches.view(patches.size(0), patches.size(1), -1)  # (B, conv_out_channels, 196)
        # Transpose to (B, 196, conv_out_channels)
        patches = patches.permute(0, 2, 1)
        # Project each patch
        patches = self.proj(patches)  # (B, 196, patch_dim)
        return patches


class QuanvolutionClassifier(nn.Module):
    """Hybrid network using the enhanced quanvolution filter followed by a GRU head."""

    def __init__(
        self,
        in_channels: int = 1,
        conv_out_channels: int = 4,
        patch_dim: int = 16,
        gru_hidden_dim: int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(
            in_channels=in_channels,
            conv_out_channels=conv_out_channels,
            patch_dim=patch_dim,
        )
        self.gru = nn.GRU(
            input_size=patch_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(gru_hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Log‑softmax over class scores, shape (B, num_classes).
        """
        # Extract patch embeddings
        patches = self.qfilter(x)  # (B, 196, patch_dim)
        # Process sequence with GRU
        _, h_n = self.gru(patches)  # h_n shape (1, B, gru_hidden_dim)
        h_n = h_n.squeeze(0)  # (B, gru_hidden_dim)
        logits = self.classifier(h_n)  # (B, num_classes)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
