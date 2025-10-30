"""Enhanced classical convolutional filter with learnable kernel, threshold, batch‑norm and dropout.

The class is a drop‑in replacement for the original Conv filter.  It adds a
* learnable 2‑D kernel (size = kernel_size)
* optional batch‑norm and dropout layers
* a trainable threshold that can be frozen or optimized jointly

The module can be used in a standard PyTorch training loop.  By exposing a
``train_step`` method the class can be used as a lightweight experiment
without the need for an external training script.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Conv(nn.Module):
    """
    Classical convolutional filter that mimics the behaviour of a quantum
    quanvolution layer while adding trainable parameters.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel.  The filter will operate on a
        ``kernel_size x kernel_size`` patch of the input.
    out_channels : int, default 1
        Number of output channels.
    padding : int, default 0
        Zero‑padding added to both sides of the input.
    threshold : float | None
        If provided, the output of the convolution is passed through a
        sigmoid and the result is shifted by ``threshold`` before the
        final activation.  This mimics the thresholding behaviour of the
        original quanvolution implementation.  ``None`` disables the
        threshold shift.
    dropout : float, default 0.0
        Dropout probability applied after the convolution.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        out_channels: int = 1,
        padding: int = 0,
        threshold: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.threshold = threshold

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels) if out_channels > 1 else None
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None

        # Optional learnable threshold parameter
        if threshold is not None:
            self.thres_param = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        else:
            self.thres_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, H, W)``.
        Returns
        -------
        torch.Tensor
            Output tensor after convolution, optional batch‑norm,
            threshold shift and dropout.
        """
        out = self.conv(x)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        if self.thres_param is not None:
            # Shift the output by the learnable threshold before sigmoid
            out = torch.sigmoid(out - self.thres_param)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    def train_step(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Perform a single training step.

        Parameters
        ----------
        data : torch.Tensor
            Batch of input data.
        targets : torch.Tensor
            Corresponding targets.
        loss_fn : nn.Module
            Loss function.
        optimizer : torch.optim.Optimizer
            Optimizer.

        Returns
        -------
        float
            Loss value for the batch.
        """
        self.train()
        optimizer.zero_grad()
        output = self(data)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, data: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the filter without training.

        Parameters
        ----------
        data : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        self.eval()
        with torch.no_grad():
            return self(data)


__all__ = ["Conv"]
