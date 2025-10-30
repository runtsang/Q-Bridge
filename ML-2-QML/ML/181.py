"""ConvGen200 – classical multi‑scale convolutional filter with attention.

This module extends the original 2×2 filter to multiple kernel sizes (2,3,4)
and learns an attention weight for each.  It can be plugged into any
PyTorch pipeline.

Example::
    from conv_gen200 import ConvGen200
    model = ConvGen200()
    out = model(torch.randn(1,1,10,10))
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["ConvGen200"]

class ConvGen200(nn.Module):
    """Multi‑scale convolutional filter with optional attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_sizes : list[int] | None
        Kernel sizes to use.  Defaults to [2, 3, 4].
    attention : bool
        Whether to use a learned attention vector over kernel sizes.
    device : str | None
        Device to place the module on.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: list[int] | None = None,
        attention: bool = True,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes or [2, 3, 4]
        self.attention = attention

        # Create a conv layer for each kernel size
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=ks,
                    bias=True,
                )
                for ks in self.kernel_sizes
            ]
        )

        # Attention weights (logits) – one per kernel size
        if self.attention:
            self.attention_logits = nn.Parameter(
                torch.zeros(len(self.kernel_sizes))
            )
        else:
            self.register_parameter("attention_logits", None)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the multi‑scale filter and combine outputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C_in, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C_out, H', W').
        """
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))

        # Stack along a new dimension: (B, C_out, H', W', K)
        stacked = torch.stack(outputs, dim=-1)

        if self.attention:
            # Compute soft‑max over kernel dimension
            attn = F.softmax(self.attention_logits, dim=0)
            # Weighted sum over kernels
            weighted = torch.einsum("...k,k->...k", stacked, attn)
            # Sum over kernel dimension
            out = weighted.sum(dim=-1)
        else:
            # Simple average over kernels
            out = stacked.mean(dim=-1)

        return out
