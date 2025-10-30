"""ConvGen275: a hybrid‑classical convolutional filter with depth‑wise separable support.

The module implements a drop‑in replacement for the original Conv class while adding:
* Multi‑channel input and output (C_in, C_out)
* Trainable kernel weights (via nn.Conv2d)
* Depth‑wise separable convolution for efficient feature extraction
* Optional quantum forward pass (see qml module)
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable

# Import the quantum module lazily to avoid heavy dependencies when not needed
try:
    from qml import ConvGen275 as QuantumConvGen275  # type: ignore
except Exception:
    QuantumConvGen275 = None  # pragma: no cover


class ConvGen275(nn.Module):
    """
    A convolutional filter that can operate in a purely classical mode or a quantum‑augmented mode.
    The default classical mode is depth‑wise separable; the quantum mode uses a parameterized
    variational circuit that we expose via the qml module.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        in_channels: int = 1,
        out_channels: int = 1,
        depthwise: bool = True,
        trainable: bool = True,
        use_quantum: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the point‑to‑point kernel.
        threshold : float
            Threshold for the bias‑like bias.
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        depthwise : bool
            If True, use depth‑wise separable convolution.
        trainable : bool
            If True, the threshold is a learnable parameter.
        use_quantum : bool
            If True, the forward pass uses the quantum circuit defined in the qml module.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = (
            nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
            if trainable
            else torch.tensor(threshold, dtype=torch.float32)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise = depthwise
        self.use_quantum = use_quantum

        if depthwise:
            # Depth‑wise separable: first depth‑wise, then point‑wise
            self.depthwise_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                groups=in_channels,
                bias=False,
            )
            self.pointwise_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=False,
            )

        if use_quantum and QuantumConvGen275 is None:
            raise ImportError(
                "Quantum backend not available. Install qiskit or pennylane."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, C_in, H, W).

        Returns
        -------
        torch.Tensor
            Scalar output per sample (mean activation).
        """
        if self.use_quantum:
            # Quantum forward pass: patch extraction and quantum evaluation
            batch_size = x.shape[0]
            outputs = []
            for i in range(batch_size):
                sample = x[i]  # shape (C_in, H, W)
                # For simplicity, we take the first channel and the first patch
                patch = sample[0, : self.kernel_size, : self.kernel_size].cpu().numpy()
                # Pad or crop to kernel_size if necessary
                if patch.shape!= (self.kernel_size, self.kernel_size):
                    patch = torch.nn.functional.interpolate(
                        sample.unsqueeze(0),
                        size=(self.kernel_size, self.kernel_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)[0].cpu().numpy()
                # Run quantum circuit
                q_output = QuantumConvGen275(
                    kernel_size=self.kernel_size,
                    threshold=self.threshold.item(),
                ).run(patch)
                outputs.append(q_output)
            return torch.tensor(outputs, dtype=torch.float32)
        else:
            if self.depthwise:
                x = self.depthwise_conv(x)
                x = self.pointwise_conv(x)
            else:
                x = self.conv(x)
            # Apply sigmoid with threshold
            activations = torch.sigmoid(x - self.threshold)
            return activations.mean(dim=[1, 2, 3])  # mean per sample


__all__ = ["ConvGen275"]
