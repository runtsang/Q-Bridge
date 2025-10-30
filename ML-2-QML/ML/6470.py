"""Enhanced classical convolutional filter with optional depthwise separable mode and learnable threshold."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Conv:
    """
    Classical convolutional filter that mimics the interface of the original
    ``Conv`` factory.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel.
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 1
        Number of output channels.
    depthwise : bool, default False
        If True, performs depthwise separable convolution (groups=in_channels).
    threshold : float, default 0.0
        Static threshold used in the sigmoid activation.
    learnable_threshold : bool, default False
        If True, the threshold becomes a trainable parameter.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        depthwise: bool = False,
        threshold: float = 0.0,
        learnable_threshold: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise = depthwise
        self.threshold = threshold
        self.learnable_threshold = learnable_threshold

        groups = in_channels if depthwise else 1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, groups=groups, bias=True
        )

        if learnable_threshold:
            # Register as a learnable parameter
            self.threshold_param = nn.Parameter(torch.tensor(threshold))
        else:
            # Store as a buffer so it is part of the state dict but not trainable
            self.register_buffer("threshold_param", torch.tensor(threshold))

    def run(self, data) -> float:
        """
        Apply the convolution and return the mean sigmoid activation.

        Parameters
        ----------
        data : array‑like or torch.Tensor
            2‑D array of shape (H, W) or (C, H, W). If a single channel is
            provided, it is automatically expanded to match the expected
            input shape.

        Returns
        -------
        float
            Mean activation value after the sigmoid.
        """
        # Convert to torch tensor
        x = torch.as_tensor(data, dtype=torch.float32)

        # Ensure shape is (batch, channel, H, W)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif x.ndim == 3 and x.shape[0] == self.in_channels:
            x = x.unsqueeze(0)  # (1, C, H, W)
        elif x.ndim == 4:
            pass  # already (N, C, H, W)
        else:
            raise ValueError("Input data must be 2‑D or 3‑D with channel dimension matching in_channels.")

        logits = self.conv(x)
        # Apply sigmoid with threshold shift
        activation = torch.sigmoid(logits - self.threshold_param)
        return activation.mean().item()


__all__ = ["Conv"]
