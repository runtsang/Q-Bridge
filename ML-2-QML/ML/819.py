"""Enhanced classical convolutional filter with multi‑scale and separable support.

This module builds on the original Conv filter but adds:
- Multiple kernel sizes (1, 2, 3) for richer receptive fields.
- Depth‑wise separable convolution to reduce parameter count.
- An L1‑style regularisation term that can be added to a loss.
- A predict method that returns a probability score and a log‑likelihood.

The class is fully compatible with PyTorch and can be dropped into any nn.Module hierarchy.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple

class ConvEnhanced(nn.Module):
    """
    Multi‑scale separable convolutional filter.

    Parameters
    ----------
    kernel_sizes : List[int]
        List of kernel sizes (e.g. [1, 2, 3]) – the module will
        generate all 1‑D convolutions that have a single channel.
    stride : int
        stride of each convolution.
    padding : int
        padding value for the input tensor.
    L1_reg : bool
        Whether to include an L1 regularisation term.
    """
    def __init__(self,
                 kernel_sizes: List[int] = [1, 2, 3],
                 stride: int = 1,
                 padding: int = 0,
                 L1_reg: bool = False) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.padding = padding
        self.L1_reg = L1_reg

        # Create a list of depth‑wise separable conv blocks
        self.blocks = nn.ModuleList()
        for k in kernel_sizes:
            # depth‑wise conv
            dw = nn.Conv2d(1, 1, kernel_size=k, stride=stride,
                           padding=padding, groups=1, bias=True)
            # point‑wise conv to mix channels (here still 1 channel)
            pw = nn.Conv2d(1, 1, kernel_size=1, bias=True)
            self.blocks.append(nn.Sequential(dw, pw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that aggregates outputs from each kernel size.
        Returns a probability score between 0 and 1.
        """
        outputs = []
        for block in self.blocks:
            out = block(x)
            out = torch.sigmoid(out)
            outputs.append(out)
        # Stack and average across kernel‑size dimension
        stacked = torch.stack(outputs, dim=0)
        mean = stacked.mean(dim=0)
        # Global average pooling to collapse spatial dims
        return mean.mean(dim=[1, 2, 3])

    def l1_loss(self) -> torch.Tensor:
        """
        Compute L1 norm of all parameters for regularisation.
        """
        if not self.L1_reg:
            return torch.tensor(0.0, device=self.blocks[0][0].weight.device)
        return torch.stack([p.abs().sum() for p in self.parameters()]).sum()

    def predict(self, x: torch.Tensor) -> Tuple[float, float]:
        """
        Return probability and log‑likelihood for a single input.
        """
        prob = self.forward(x).item()
        log_likelihood = torch.log(torch.tensor(prob + 1e-12)).item()
        return prob, log_likelihood
