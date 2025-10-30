"""HybridSamplerConv: classical hybrid sampler with convolutional preprocessing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """
    Classical 2D convolutional filter emulating a quantum quanvolution layer.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor) -> float:
        """
        Forward pass: applies convolution, sigmoid activation, and returns mean activation.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class SamplerModule(nn.Module):
    """
    Classical sampler network mirroring the QNN helper.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, output_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridSamplerConv(nn.Module):
    """
    Hybrid model combining a convolutional pre‑processor and a sampler network.
    """
    def __init__(self,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 sampler_input_dim: int = 2,
                 sampler_hidden_dim: int = 4,
                 sampler_output_dim: int = 2) -> None:
        super().__init__()
        self.conv = ConvFilter(conv_kernel_size, conv_threshold)
        self.sampler = SamplerModule(sampler_input_dim, sampler_hidden_dim, sampler_output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolutional filter to the input data and feed the scalar output
        into the sampler network. The scalar is duplicated to match the sampler input shape.
        """
        conv_out = self.conv.run(x)
        sampler_input = torch.tensor([conv_out, conv_out], dtype=torch.float32)
        return self.sampler(sampler_input)

__all__ = ["HybridSamplerConv"]
