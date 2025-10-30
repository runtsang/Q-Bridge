"""Hybrid classical network that mirrors quantum modules for easy comparison.

This module defines a single ``HybridNet`` class that internally uses
classical counterparts of the quantum
fully‑connected layer (FCL), quanvolution filter (Conv),
and SamplerQNN.  The design allows a drop‑in replacement
for the quantum implementation, keeping the same API.

The class inherits from ``torch.nn.Module`` and exposes a
``forward`` method that returns the outputs of the three
sub‑modules.  All sub‑modules are defined as functions
to stay consistent with the original seed structure.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

__all__ = ["HybridNet"]


def FCL():
    """Return a classical fully‑connected layer mimicking the quantum FCL."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            # Linear mapping to a single output
            self.linear = nn.Linear(n_features, 1)

        def forward(self, thetas: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            thetas: torch.Tensor
                Shape (batch, n_features).  The first element is treated
                as the quantum parameter in the original example.
            Returns
            -------
            torch.Tensor
                Mean tanh activation over the batch.
            """
            values = thetas.float()
            out = self.linear(values)
            return torch.tanh(out).mean(dim=0, keepdim=True)

    return FullyConnectedLayer()


def Conv(kernel_size: int = 2, threshold: float = 0.0):
    """Return a classical 2‑D convolution that emulates a quanvolution filter."""
    class ConvFilter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                bias=True,
            )

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            data: torch.Tensor
                Shape (H, W) or (N, H, W) where N is batch size.
            Returns
            -------
            torch.Tensor
                Mean sigmoid activation after convolution.
            """
            if data.ndim == 2:
                data = data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            else:
                data = data.unsqueeze(1)  # (N,1,H,W)
            logits = self.conv(data)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean(dim=(2, 3))

    return ConvFilter()


def SamplerQNN(input_dim: int = 2, output_dim: int = 2):
    """Return a simple sampling network that mimics the quantum SamplerQNN."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 4),
                nn.Tanh(),
                nn.Linear(4, output_dim),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            inputs: torch.Tensor
                Shape (batch, input_dim).
            Returns
            -------
            torch.Tensor
                Soft‑max probabilities over the output dimension.
            """
            return torch.softmax(self.net(inputs), dim=-1)

    return SamplerModule()


class HybridNet(nn.Module):
    """Hybrid classical network that combines FCL, Conv, and SamplerQNN."""
    def __init__(
        self,
        n_features: int = 1,
        conv_kernel: int = 2,
        sampler_input_dim: int = 2,
    ) -> None:
        super().__init__()
        self.fcl = FCL()(n_features)
        self.conv = Conv(kernel_size=conv_kernel)()
        self.sampler = SamplerQNN(input_dim=sampler_input_dim)()

    def forward(
        self,
        thetas: torch.Tensor,
        image: torch.Tensor,
        sampler_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the hybrid pipeline.

        Parameters
        ----------
        thetas: torch.Tensor
            Values for the fully‑connected layer.
        image: torch.Tensor
            2‑D image or batch of images for convolution.
        sampler_input: torch.Tensor
            Inputs for the sampler network.

        Returns
        -------
        tuple
            (fcl_out, conv_out, sampler_out)
        """
        fcl_out = self.fcl(thetas)
        conv_out = self.conv(image)
        sampler_out = self.sampler(sampler_input)
        return fcl_out, conv_out, sampler_out
