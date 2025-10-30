"""Quantum implementation of the QuanvolutionHybrid model.

The network applies a 2×2 convolution to extract image patches, then
encodes each patch into a four‑qubit register using a Ry rotation per
input value.  A RandomLayer injects trainable noise, the state is
measured in the Pauli‑Z basis, and the resulting bit‑strings form
features for a linear head.  The module supports both classification
and regression with a shared interface.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum hybrid model for image classification or regression.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input image channels.
    out_features : int, default 10
        Number of output classes (ignored when ``regression=True``).
    kernel_size : int, default 2
        Size of the convolution kernel that creates image patches.
    stride : int, default 2
        Stride of the convolution kernel.
    num_wires : int, default 4
        Number of qubits used to encode each patch.
    regression : bool, default False
        If ``True`` the head outputs a single regression value
        instead of class logits.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 10,
        kernel_size: int = 2,
        stride: int = 2,
        num_wires: int = 4,
        regression: bool = False,
    ) -> None:
        super().__init__()
        self.regression = regression
        self.conv = nn.Conv2d(in_channels, num_wires, kernel_size=kernel_size, stride=stride)

        # Quantum encoder: Ry rotation per input value
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(num_wires)]
        )
        # Random layer to increase expressivity
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, out_features if not regression else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Log‑probabilities for classification or a regression output.
        """
        patches = self.conv(x)  # (B, num_wires, H', W')
        B, W, H, W_ = patches.shape
        patches = patches.view(B, W, -1).transpose(1, 2)  # (B, N, W)
        N = patches.shape[1]

        # Construct a batch of quantum devices for all patches
        qdev = tq.QuantumDevice(n_wires=W, bsz=B * N, device=patches.device)

        # Flatten patches for encoding
        flat_patches = patches.reshape(-1, W)  # (B·N, W)
        self.encoder(qdev, flat_patches)
        self.random_layer(qdev)
        measured = self.measure(qdev)  # (B·N, W)

        # Reshape and pool over patches
        features = measured.view(B, N, -1).mean(dim=1)  # (B, W)
        logits = self.head(features)  # (B, out_features) or (B, 1)

        if not self.regression:
            return F.log_softmax(logits, dim=-1)
        return logits


__all__ = ["QuanvolutionHybrid"]
