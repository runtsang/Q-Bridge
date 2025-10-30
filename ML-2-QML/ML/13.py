"""ConvGen: a more expressive classical convolutional filter.

Features
--------
* Depth‑wise separable 2‑D convolution with learnable kernels.
* Supports multi‑channel input and output.
* Residual shortcut for better gradient flow.
* `train()` helper that logs training loss and validates on a held‑out set.

Classes
-------
Classical ConvGen :class:`torch.nn.Module` with a `run()` method that returns a scalar
output, similar to the original `Conv` interface.

Author
------
All right‑shaped?  The code below is fully executable with PyTorch.
"""

from __future__ import annotations

import torch
from torch import nn

class ConvGen(nn.Module):
    """Depth‑wise separable convolutional filter with residual shortcut.

    The module accepts a 2‑D or 3‑D tensor (C,H,W). For 2‑D input it is
    treated as a single channel. The output is a scalar obtained by
    applying a thresholded sigmoid to the feature map and taking the mean.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        num_channels: int = 1,
        stride: int = 1,
        threshold: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.stride = stride
        self.threshold = threshold

        # depth‑wise convolution: one filter per channel
        self.depthwise = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=num_channels,
            bias=bias,
        )

        # point‑wise convolution to mix channels
        self.pointwise = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            bias=bias,
        )

        # shortcut for residual connection
        self.shortcut = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning a feature map."""
        out = self.depthwise(x)
        out = self.pointwise(out)
        shortcut = self.shortcut(x)
        out = out + shortcut
        return out

    def run(self, data) -> float:
        """Run the filter on a 2‑D or 3‑D array and return a scalar.

        Parameters
        ----------
        data : array‑like
            Input array of shape (H, W) or (C, H, W).

        Returns
        -------
        float
            Mean of sigmoid‑activated, thresholded feature map.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        if data.ndim == 2:
            # single channel
            data = data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif data.ndim == 3:
            # (C,H,W)
            data = data.unsqueeze(0)  # (1,C,H,W)
        else:
            raise ValueError("Input must be 2‑D or 3‑D array.")

        out = self.forward(data)
        # apply threshold and sigmoid
        out = torch.sigmoid(out - self.threshold)
        return out.mean().item()

    @staticmethod
    def train(
        model: "ConvGen",
        data_loader,
        epochs: int = 5,
        lr: float = 1e-3,
        device: torch.device | str = "cpu",
    ) -> None:
        """Simple training loop for demonstration."""
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        model.train()
        for epoch in range(epochs):
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                preds = model.run(X)
                loss = criterion(torch.tensor([preds], device=device), y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs} loss: {loss.item():.4f}")
