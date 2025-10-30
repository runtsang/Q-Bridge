"""ConvGen: a multi‑scale, learnable convolutional filter for classical and quantum experiments.

The module defines a shared class name `ConvGen` that can be instantiated with either a
`kernel_sizes` list or a single size.  A depth‑wise separable convolution is
performed for each kernel size and the outputs are fused by a trainable
linear layer.  The design mirrors the original Conv.py but adds
* multi‑scale support
* a `trainable` flag that toggles the learnability of the kernel
* a small helper that returns the mean activation for a single sample
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np


class ConvGen(nn.Module):
    def __init__(
        self,
        kernel_sizes: list[int] | int = 2,
        threshold: float | None = None,
        trainable: bool = True,
        depthwise: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        kernel_sizes : list or int
            If a single integer, the same kernel size will be applied
            to all channels.  If a list, each element specifies a
            separate 2‑D filter that will be applied depth‑wise.
        threshold : float or None
            The threshold value used in the quantum‑like activation.
            If None, the sigmoid‑based activation is used.
        trainable : bool
            Whether the convolutional kernels are learnable.
        depthwise : bool
            Whether to use depth‑wise separable convolutions.
        """
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = kernel_sizes
        self.threshold = threshold
        self.trainable = trainable
        self.depthwise = depthwise

        # Create a separate conv layer for each kernel size
        self.convs = nn.ModuleList()
        for ks in self.kernel_sizes:
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=ks,
                bias=True,
                groups=1 if not depthwise else 1,
            )
            if not trainable:
                conv.weight.requires_grad = False
                conv.bias.requires_grad = False
            self.convs.append(conv)

        # Fusion layer: linear combination of per‑kernel outputs
        self.fusion = nn.Linear(len(self.kernel_sizes), 1, bias=False)
        if not trainable:
            self.fusion.weight.requires_grad = False

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape (batch, 1, H, W) where H and W are at least
            as large as the largest kernel in `kernel_sizes`.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 1) containing the fused activation
            for each sample in the batch.
        """
        batch = data.shape[0]
        outputs = []
        for idx, ks in enumerate(self.kernel_sizes):
            # Extract top‑left sub‑window of size ks×ks
            sub = data[:, :, :ks, :ks]
            logits = self.convs[idx](sub)
            if self.threshold is None:
                act = torch.sigmoid(logits)
            else:
                act = torch.sigmoid(logits - self.threshold)
            outputs.append(act.mean(dim=(2, 3)))  # mean over spatial dims

        # Stack per‑kernel outputs: shape (batch, num_kernels)
        stacked = torch.stack(outputs, dim=1)
        fused = self.fusion(stacked)  # shape (batch, 1)
        return fused

    def mean_activation(self, data: np.ndarray) -> float:
        """
        Compute the mean activation for a single sample.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (H, W).

        Returns
        -------
        float
            Mean activation value.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return self.forward(tensor).item()

__all__ = ["ConvGen"]
