"""Hybrid classical convolutional filter with a learnable linear patch transformation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybridClassifier(nn.Module):
    """
    Classical implementation of a quanvolutional filter followed by a linear classifier.
    The filter extracts 2x2 patches, applies a learnable linear transformation per patch,
    then the features are flattened and passed to a fully connected head.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, patch_size: int = 2, stride: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        # Convolution that extracts 2x2 patches and produces 4 feature maps
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=patch_size, stride=stride)
        # Optional linear layer to refine patch features (learnable)
        self.patch_linear = nn.Linear(4, 4)
        self.head = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: input tensor of shape (batch, channels, height, width)
        :return: log probabilities of shape (batch, num_classes)
        """
        # Extract patches
        patches = self.conv(x)  # shape: (batch, 4, 14, 14)
        # Flatten spatial dimensions
        patches = patches.view(patches.size(0), 4, -1)  # (batch, 4, 196)
        # Apply linear transformation to each patch
        patches = self.patch_linear(patches.permute(0, 2, 1))  # (batch, 196, 4)
        patches = patches.permute(0, 2, 1).contiguous()  # (batch, 4, 196)
        # Flatten all features
        features = patches.view(patches.size(0), -1)  # (batch, 4*196)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)

    def set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility of initial weights.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

__all__ = ["QuanvolutionHybridClassifier"]
