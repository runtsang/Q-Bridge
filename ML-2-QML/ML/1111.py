"""Enhanced classical convolution filter with multi‑scale, depthwise separable, and adaptive threshold.

The public API mirrors the original: Conv() returns a callable object with a run(data) method.
"""

from __future__ import annotations

import torch
from torch import nn

class ConvEnhanced(nn.Module):
    """
    Multi‑scale depthwise separable convolution filter.

    Parameters
    ----------
    kernel_sizes : list[int]
        List of kernel sizes to apply in parallel. Defaults to [1, 3].
    threshold : float | None
        If None, threshold is learned as a parameter; otherwise fixed.
    depthwise : bool
        Whether to use depthwise separable convolution.
    """

    def __init__(
        self,
        kernel_sizes: list[int] = [1, 3],
        threshold: float | None = None,
        depthwise: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.depthwise = depthwise

        # Learnable or fixed threshold
        if threshold is None:
            self.threshold = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            self.threshold = torch.tensor(threshold, dtype=torch.float32)

        # Build convolution layers for each kernel size
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            if depthwise:
                # depthwise separable: 1x1 conv followed by depthwise conv
                self.convs.append(
                    nn.Sequential(
                        nn.Conv2d(1, 1, kernel_size=1, bias=False),
                        nn.Conv2d(1, 1, kernel_size=k, padding=k // 2, bias=True),
                    )
                )
            else:
                self.convs.append(nn.Conv2d(1, 1, kernel_size=k, padding=k // 2, bias=True))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the filter response for a single 2‑D patch.

        Parameters
        ----------
        data : torch.Tensor
            2‑D array of shape (kernel_size, kernel_size) or larger.

        Returns
        -------
        torch.Tensor
            Scalar activation.
        """
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        else:
            raise ValueError("Input must be 2‑D array")

        outputs = []
        for conv in self.convs:
            out = conv(data)
            out = torch.sigmoid(out - self.threshold)
            outputs.append(out.mean())
        return torch.stack(outputs).mean()

    def run(self, data):
        """Compatibility wrapper for the original API."""
        return self.forward(data).item()

def Conv(**kwargs):
    """
    Factory function to create a ConvEnhanced instance.

    Accepts the same keyword arguments as ConvEnhanced.
    """
    return ConvEnhanced(**kwargs)
