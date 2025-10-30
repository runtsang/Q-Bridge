"""Unified classical implementation of a quantum-inspired convolution and sampler network.

Combines the ConvFilter and SamplerModule from the original seeds into a single
class that can be used as a drop‑in replacement for the quantum Conv and SamplerQNN
implementations. The class exposes a `forward` method compatible with PyTorch
modules and a `run` helper that returns a scalar probability from the
convolution filter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class QuantumConvSampler(nn.Module):
    """
    Classical counterpart of a quantum convolution + sampler network.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        sampler_hidden: int = 4,
        sampler_output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold

        # Convolutional filter
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Sampler network
        self.sampler = nn.Sequential(
            nn.Linear(2, sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, sampler_output_dim),
        )

    def conv_run(self, data: Any) -> float:
        """
        Run the 2‑D convolution filter on the input array.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean sigmoid activation after thresholding.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.conv_threshold)
        return activations.mean().item()

    def forward(self, data: Any) -> torch.Tensor:
        """
        Forward pass that emulates the quantum circuit.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Output of the sampler network (softmax probabilities).
        """
        conv_val = self.conv_run(data)
        inputs = torch.tensor([conv_val, conv_val], dtype=torch.float32)
        return F.softmax(self.sampler(inputs), dim=-1)

    def run(self, data: Any) -> float:
        """
        Convenience wrapper that returns only the scalar probability from the
        convolution step, matching the signature of the original quantum filter.
        """
        return self.conv_run(data)


__all__ = ["QuantumConvSampler"]
