"""ConvHybrid: classical backbone with optional variational integration."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable

class ConvHybrid(nn.Module):
    """
    Classical depth‑wise separable convolutional filter with an optional
    variational block that can be swapped with a real quantum circuit.

    Parameters
    ----------
    input_channels : int, default 1
        Number of input channels.
    kernel_sizes : Iterable[int], default (1, 3, 5)
        Kernel sizes for the depth‑wise convolutions.
    threshold : float, default 0.0
        Threshold for the sigmoid activation.
    use_variational : bool, default False
        If True, a dummy 1×1 variational layer is added and the final
        output is a gated combination of classical and variational logits.
    """

    def __init__(
        self,
        input_channels: int = 1,
        kernel_sizes: Iterable[int] = (1, 3, 5),
        threshold: float = 0.0,
        use_variational: bool = False,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.use_variational = use_variational

        # Depth‑wise separable convolutions
        self.depthwise = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2
            self.depthwise.append(
                nn.Conv2d(
                    input_channels,
                    input_channels,
                    kernel_size=k,
                    padding=pad,
                    groups=input_channels,
                )
            )
        # Point‑wise convolution to mix scales
        self.pointwise = nn.Conv2d(
            input_channels * len(kernel_sizes),
            4,
            kernel_size=1,
        )
        # Gating layer (scalar gate for each batch item)
        self.gate = nn.Conv2d(
            4,
            1,
            kernel_size=1,
        )
        # Dummy variational layer (can be replaced by a quantum circuit)
        if self.use_variational:
            self.variational = nn.Conv2d(
                input_channels,
                1,
                kernel_size=1,
            )
        else:
            self.variational = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Scalar probability per batch item.
        """
        # Depth‑wise separable block
        dw_out = [dw(x) for dw in self.depthwise]
        dw_concat = torch.cat(dw_out, dim=1)
        pw_out = self.pointwise(dw_concat)

        # Classical activation
        cls = torch.sigmoid(pw_out - self.threshold)
        cls_mean = cls.mean(dim=(2, 3))  # (B, 4)

        if self.use_variational and self.variational is not None:
            var = self.variational(x)
            var_mean = torch.sigmoid(var - self.threshold).mean(dim=(2, 3))  # (B, 1)
            # Obtain a scalar gate per batch item
            gate = torch.sigmoid(self.gate(pw_out).mean(dim=(2, 3)))  # (B, 1)
            out = gate * cls_mean + (1 - gate) * var_mean
            out = out.mean(dim=1)  # collapse channel dimension
        else:
            out = cls_mean.mean(dim=1)  # collapse channel dimension
        return out.squeeze()

    def run(self, data) -> float:
        """
        Convenience wrapper to run on a single data patch.

        Parameters
        ----------
        data : array‑like or torch.Tensor
            Shape (C, H, W) or (H, W) if single channel.

        Returns
        -------
        float
            Probability value.
        """
        if isinstance(data, torch.Tensor):
            tensor = data.unsqueeze(0) if data.ndim == 3 else data
        else:
            tensor = torch.tensor(data, dtype=torch.float32)
            if tensor.ndim == 2:  # single channel
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            return self.forward(tensor).item()

def Conv() -> ConvHybrid:
    """
    Drop‑in factory function compatible with the original API.

    Returns
    -------
    ConvHybrid
        An instance of the enhanced filter.
    """
    return ConvHybrid()

__all__ = ["ConvHybrid", "Conv"]
