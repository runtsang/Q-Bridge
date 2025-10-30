"""ConvGen258: classical convolution filter with multi‑kernel and depthwise support."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = ["ConvGen258"]


class ConvGen258(nn.Module):
    """
    Classical convolution filter that can process multiple kernel sizes and optionally
    operate in depthwise mode. The filter is drop‑in compatible with the original
    Conv class but adds richer functionality.

    Parameters
    ----------
    kernel_sizes : list[int] or tuple[int,...], optional
        The kernel sizes to apply.  Defaults to ``[2]``.
    depthwise : bool, default False
        Whether to use depthwise separable convolutions.  When ``True`` the
        convolution is applied per channel; for a single channel this is a no‑op
        but keeps the interface consistent.
    threshold : float, default 0.0
        Value subtracted from the convolution logits before sigmoid activation.
    """

    def __init__(
        self,
        kernel_sizes: list[int] | tuple[int,...] | None = None,
        depthwise: bool = False,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes or [2]
        self.depthwise = depthwise
        self.threshold = threshold

        # Build a module for each kernel size
        self.convs = nn.ModuleList()
        for k in self.kernel_sizes:
            self.convs.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=k,
                    bias=True,
                    groups=1 if not depthwise else 1,
                )
            )

        self.activation = nn.Sigmoid()

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        """
        Apply the filter to 2‑D data and return the mean activation.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            Array of shape (H, W) or (N, H, W).  The data is cast to a torch
            tensor internally.

        Returns
        -------
        float
            Mean sigmoid activation over all kernels and spatial locations.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (N=1, C=1, H, W)
        elif data.ndim == 3:
            # (N, H, W) -> (N, C=1, H, W)
            data = data.unsqueeze(1)
        else:
            raise ValueError("Input must be 2‑D or 3‑D array")

        outputs = []
        for conv in self.convs:
            logits = conv(data)  # shape (N, 1, H', W')
            logits = logits - self.threshold
            act = self.activation(logits)
            outputs.append(act.mean(dim=[2, 3]))

        # Concatenate across kernel sizes and average
        out = torch.cat(outputs, dim=-1).mean(dim=-1).mean()
        return out.item()

    def forward(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Forward method that returns a torch tensor of per‑sample activations.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            data = data.unsqueeze(1)
        else:
            raise ValueError("Input must be 2‑D or 3‑D array")

        outputs = []
        for conv in self.convs:
            logits = conv(data)
            logits = logits - self.threshold
            act = self.activation(logits)
            outputs.append(act.mean(dim=[2, 3]))

        out = torch.cat(outputs, dim=-1).mean(dim=-1)
        return out.squeeze(1)
