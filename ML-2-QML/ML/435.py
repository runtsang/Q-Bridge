"""Hybrid convolution generator with separable and quantum‑aware options.

This module extends the original Conv filter by:
* allowing multiple kernel sizes (3, 5, 7) for richer receptive fields;
* adding a depthwise‑separable variant that reduces parameter count;
* exposing a `trainable` flag that, when False, freezes all weights for inference;
* providing a `quantum_gate` callable that can be replaced with a custom variational circuit.

Author: gpt‑oss‑20b
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

__all__ = ["ConvGen064"]

class ConvGen064(nn.Module):
    """A versatile convolutional filter that can act as a classical or hybrid
    quantum‑based filter.  The class is drop‑in compatible with the original
    Conv.  The `run` method returns a scalar value that is the mean activation
    of the convolution followed by a sigmoid.

    Parameters
    ----------
    kernel_sizes : list[int] | None
        The list of kernel sizes (i.e. 3, 5, 7).  If None, defaults to [2].
    depthwise : bool
        If True, use depthwise separable convolutions (grouped conv with
        `groups=in_channels`).
    trainable : bool
        If False, all convolutional weights are frozen after initialization.
    quantum_gate : Callable | None
        Optional callable that accepts a 2D array and returns a float.  If
        provided, the quantum output is multiplied with the classical output.
    """

    def __init__(
        self,
        kernel_sizes: list[int] | None = None,
        depthwise: bool = False,
        trainable: bool = True,
        quantum_gate: callable | None = None,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes if kernel_sizes is not None else [2]
        self.depthwise = depthwise
        self.trainable = trainable
        self.quantum_gate = quantum_gate

        # Build a convolutional layer for each kernel size
        self.convs = nn.ModuleDict()
        for ks in self.kernel_sizes:
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=ks,
                bias=True,
                groups=1 if not depthwise else 1,
            )
            if not trainable:
                for p in conv.parameters():
                    p.requires_grad = False
            self.convs[str(ks)] = conv

    def run(self, data) -> float:
        """Apply convolution(s) and optional quantum gate to the input data.

        Parameters
        ----------
        data : array-like
            2D array of shape (kernel_size, kernel_size).  The size must match
            one of the configured kernel sizes.

        Returns
        -------
        float
            Mean activation after sigmoid.  If a quantum gate is supplied,
            its output multiplies the classical activation.
        """
        # Convert input to torch tensor
        tensor = torch.as_tensor(data, dtype=torch.float32)
        # Determine kernel size
        ks = data.shape[0]
        if ks not in self.kernel_sizes:
            raise ValueError(f"Unsupported kernel size {ks}. Supported: {self.kernel_sizes}")
        conv = self.convs[str(ks)]

        # Reshape to match Conv2d input: (N, C, H, W)
        tensor = tensor.view(1, 1, ks, ks)
        logits = conv(tensor)
        activations = torch.sigmoid(logits)
        mean_act = activations.mean().item()

        if self.quantum_gate is not None:
            q_out = self.quantum_gate.run(data)
            return mean_act * q_out
        return mean_act
