"""Purely classical implementation of the extended quanvolution module.

The class `QuanvolutionHybrid` mirrors the interface of the original
Quanvolution example but incorporates:
  * a depth‑wise 2×2 convolutional filter that outputs `4 * depth` channels,
  * a shortcut 1×1 convolution that bypasses the filter,
  * optional dropout for regularisation,
  * a simple linear head and log‑softmax output.

The module exposes a ``pretrain_contrastive`` method that returns the
features before the linear head; this can be used to pre‑train the
filter with a contrastive loss without touching the rest of the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """
    Classical depth‑wise quanvolution module with shortcut and contrastive
    pre‑training hook.

    Parameters
    ----------
    depth : int, default=2
        Number of 2×2 convolutional “depths” (output channels per depth are
        four).  The total number of output channels is ``4 * depth``.
    dropout : float, default=0.1
        Dropout probability applied to the concatenated feature map before
        the linear head.
    """

    def __init__(self, depth: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.depth = depth
        # 2×2 depth‑wise filter producing 4 channels per depth
        self.qfilter = nn.Conv2d(
            in_channels=1,
            out_channels=4 * depth,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        # Shortcut: 1×1 conv to match channel dimension
        self.shortcut = nn.Conv2d(
            in_channels=1,
            out_channels=4 * depth,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(4 * depth * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log softmax logits over 10 classes.
        """
        # Main quantum‑like branch
        main = self.qfilter(x).view(x.size(0), -1)
        # Shortcut branch
        skip = self.shortcut(x).view(x.size(0), -1)
        # Combine
        features = main + skip
        features = self.dropout(features)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

    def pretrain_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the feature vector before the linear head.  This can be fed
        to a contrastive loss (e.g. NT-Xent) for unsupervised pre‑training.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (batch, 4*depth*14*14).
        """
        main = self.qfilter(x).view(x.size(0), -1)
        skip = self.shortcut(x).view(x.size(0), -1)
        return main + skip

__all__ = ["QuanvolutionHybrid"]
