"""Classical hybrid model with a learnable 2×2 convolution and residual connections.

The model replaces the random quanvolution kernel with a trainable 2×2
convolutional filter and a lightweight residual block.  The output is
flattened and passed through a linear classifier.  The interface
(`forward`) matches the original seed, accepting a batch of 28×28
single‑channel images and returning log‑softmax logits over ten
classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionClassifier(nn.Module):
    """
    Classical CNN that emulates the structure of the original
    quanvolution filter but with learnable parameters.
    """

    def __init__(self) -> None:
        super().__init__()
        # Learnable 2×2 convolutional filter
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(4)

        # Residual block: a 1×1 conv + BN
        self.res_conv = nn.Conv2d(4, 4, kernel_size=1, bias=False)
        self.res_bn = nn.BatchNorm2d(4)

        # Linear head
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, 10).
        """
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        # Residual addition
        res = self.res_conv(out)
        res = self.res_bn(res)
        out = out + res
        out = F.relu(out)

        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionClassifier"]
