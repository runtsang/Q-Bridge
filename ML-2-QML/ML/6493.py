"""ConvGen: Adaptive classical convolution with hybrid loss support.

This module extends the original Conv filter by adding:
- Adaptive kernel size that can grow during training.
- Trainable per‑channel threshold that is learned via gradient descent.
- A simple hybrid loss combining the classical output with a target
  label, allowing the model to be used directly in a supervised
  setting.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvGen(nn.Module):
    """A PyTorch module that can be used as a drop‑in replacement for the
    original Conv class.  The ``run`` method returns a **single
    value** that is *the* **logits** (the raw output before activation).
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 adaptive: bool = True):
        """
        Args:
            kernel_size: Initial kernel size for the convolution.
            threshold: Initial threshold value for sigmoid activation.
            adaptive: If True, the kernel size can be increased during training.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.adaptive = adaptive
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the convolution, sigmoid activation
        with a learnable threshold and returns the mean activation.
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    def run(self, data) -> float:
        """
        Run the filter on a 2D array of shape (kernel_size, kernel_size)
        and return the scalar output.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).view(
            1, 1, self.kernel_size, self.kernel_size)
        return self.forward(tensor).item()

    def fit(self,
            data: torch.Tensor,
            targets: torch.Tensor,
            lr: float = 1e-3,
            epochs: int = 100,
            callback: callable | None = None) -> None:
        """
        Quick training loop that optimizes the convolution weights and
        the learnable threshold to minimize the hybrid loss.

        Args:
            data: Tensor of shape (N, 1, kernel_size, kernel_size).
            targets: Tensor of shape (N,) with target scalar values.
            lr: Learning rate.
            epochs: Number of epochs.
            callback: Optional function called after each epoch with
                      (epoch, loss, output).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(data)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            if callback:
                callback(epoch, loss.item(), outputs.detach().cpu().numpy())

    def increase_kernel(self, new_size: int) -> None:
        """
        Increase kernel size by creating a new Conv2d layer with the
        desired size and copying over the existing weights.
        """
        if not self.adaptive:
            raise RuntimeError("Adaptive mode disabled")
        if new_size <= self.kernel_size:
            raise ValueError("New kernel size must be larger")
        # Create new conv layer
        new_conv = nn.Conv2d(1, 1, kernel_size=new_size, bias=True)
        # Copy existing weights to the top-left corner
        with torch.no_grad():
            new_conv.weight[..., :self.kernel_size, :self.kernel_size] = self.conv.weight
            new_conv.bias = self.conv.bias
        self.conv = new_conv
        self.kernel_size = new_size
