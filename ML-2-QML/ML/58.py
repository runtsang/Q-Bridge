"""
Enhanced classical convolution module with trainable weights and optional depthwise separable filtering.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Conv(nn.Module):
    """
    A drop‑in replacement for the original Conv filter.
    Supports:
        * Standard convolution (default).
        * Depth‑wise separable convolution when ``depthwise=True``.
        * Optional bias‑free mode.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        depthwise: bool = False,
        bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.depthwise = depthwise
        self.bias = bias

        if depthwise:
            self.depthwise_conv = nn.Conv2d(
                1,
                1,
                kernel_size=kernel_size,
                groups=1,
                bias=bias,
                padding=0,
            )
            self.pointwise_conv = nn.Conv2d(
                1,
                1,
                kernel_size=1,
                bias=bias,
                padding=0,
            )
        else:
            self.conv = nn.Conv2d(
                1,
                1,
                kernel_size=kernel_size,
                bias=bias,
                padding=0,
            )

        # Initialize weights
        if kernel_initializer == "glorot_uniform":
            nn.init.xavier_uniform_(self.conv.weight)
        elif kernel_initializer == "he_normal":
            nn.init.kaiming_normal_(self.conv.weight)

        if depthwise:
            nn.init.xavier_uniform_(self.depthwise_conv.weight)
            nn.init.xavier_uniform_(self.pointwise_conv.weight)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution to the input data.

        Args:
            data: Tensor of shape (batch, 1, H, W) or (1, H, W).

        Returns:
            Tensor of the same shape as the input after convolution and sigmoid activation.
        """
        if data.ndim == 3:
            data = data.unsqueeze(0)  # Add batch dimension

        if self.depthwise:
            out = self.depthwise_conv(data)
            out = self.pointwise_conv(out)
        else:
            out = self.conv(data)

        out = torch.sigmoid(out - self.threshold)
        return out

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that applies the forward pass and returns a scalar
        mean activation value.

        Args:
            data: 2D array or torch tensor of shape (H, W).

        Returns:
            Average activation as a float.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        activations = self.forward(tensor)
        return activations.mean().item()

    def train(self, data_loader, optimizer, loss_fn, epochs: int = 10):
        """
        Train the Conv layer on a data loader.

        Args:
            data_loader: Iterable yielding (input, target) pairs.
            optimizer: Torch optimizer.
            loss_fn: Loss function.
            epochs: Number of training epochs.
        """
        self.train()
        for epoch in range(epochs):
            for x, y in data_loader:
                optimizer.zero_grad()
                y_pred = self.forward(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
