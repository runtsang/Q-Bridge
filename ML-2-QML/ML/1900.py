"""Classical quanvolution network with depthwise separable convolution.

The network replaces the simple 2‑pixel filter of the seed with a 1×1
projection followed by a 2×2 unfolding.  The resulting patch
features are flattened and passed through a dropout‑regularised linear
head.  The design is fully compatible with PyTorch’s training loop
and can serve as a strong classical baseline for ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionNet(nn.Module):
    """Depthwise separable quanvolution for MNIST‑style images.

    Parameters
    ----------
    in_channels
        Number of input image channels (default 1 for MNIST).
    hidden_channels
        Output channels of the 1×1 projection.  Larger values give a
        richer patch representation.
    num_classes
        Number of classification targets.
    dropout
        Dropout probability applied after flattening.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        # 2×2 patches with stride 2 → 14×14 = 196 patches
        self.unfold = nn.Unfold(kernel_size=2, stride=2)
        # input dim = hidden_channels * 4 * 196
        self.linear = nn.Linear(hidden_channels * 4 * 14 * 14, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log‑softmax logits.

        Parameters
        ----------
        x
            Input tensor of shape (B, C, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax probabilities of shape (B, num_classes).
        """
        x = self.preprocess(x)
        patches = self.unfold(x)  # (B, C*4, 196)
        patches = patches.transpose(1, 2)  # (B, 196, C*4)
        flat = patches.reshape(patches.size(0), -1)  # (B, C*4*196)
        flat = self.dropout(flat)
        logits = self.linear(flat)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionNet"]
