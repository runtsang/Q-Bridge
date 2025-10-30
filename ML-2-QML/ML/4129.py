"""Hybrid estimator with classical conv, sampler, and regressor.

The module contains:
- ConvFilter: a 2‑D convolution that emulates a quantum filter.
- SamplerModule: a softmax sampler producing a 2‑dimensional probability vector.
- EstimatorNet: a fully‑connected network whose input dimension adapts to the chosen sub‑modules.
- EstimatorQNNGen104: a torch.nn.Module that stitches the components together.

All components are fully differentiable and can be trained jointly with standard PyTorch optimizers.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["ConvFilter", "SamplerModule", "EstimatorNet", "EstimatorQNNGen104"]


class ConvFilter(nn.Module):
    """Emulates a quantum filter using a 2‑D convolution.

    The filter operates on 2×2 input patches and returns a single scalar
    activation.  The threshold controls a sigmoid activation that
    mimics the probabilistic nature of a quantum measurement.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        """Apply the filter to a 2×2 patch.

        Args:
            data: Array of shape (2, 2).

        Returns:
            float: mean sigmoid activation.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


class SamplerModule(nn.Module):
    """Softmax sampler that maps features to a 2‑dimensional probability vector."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


class EstimatorNet(nn.Module):
    """Fully‑connected regression network with a dynamic input size."""

    def __init__(self, input_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class EstimatorQNNGen104(nn.Module):
    """Hybrid estimator that optionally uses a conv filter, a sampler, and a quantum layer.

    Parameters
    ----------
    use_conv : bool
        Whether to prepend the ConvFilter.
    use_sampler : bool
        Whether to prepend the SamplerModule.
    use_quantum : bool
        Whether to attach a quantum circuit (currently only a placeholder).
    """

    def __init__(
        self,
        use_conv: bool = True,
        use_sampler: bool = True,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.use_conv = use_conv
        self.use_sampler = use_sampler
        self.use_quantum = use_quantum

        # Build sub‑modules
        if self.use_conv:
            self.conv = ConvFilter(kernel_size=2, threshold=0.0)
        if self.use_sampler:
            self.sampler = SamplerModule()

        # Compute dynamic input dimension for the regressor
        input_dim = 2  # base features
        if self.use_conv:
            input_dim += 1
        if self.use_sampler:
            input_dim += 1
        self.estimator = EstimatorNet(input_dim=input_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (batch, 2). The first two columns are
                    the base features; the remaining columns are optional
                    conv and sampler outputs.

        Returns:
            Tensor of shape (batch, 1) – regression prediction.
        """
        batch_size = inputs.shape[0]
        # Base features
        features = inputs

        # Conv output
        if self.use_conv:
            conv_out = torch.empty(batch_size, 1, device=inputs.device, dtype=inputs.dtype)
            for i in range(batch_size):
                patch = inputs[i].cpu().numpy().reshape(2, 2)
                conv_out[i] = self.conv.run(patch)
            features = torch.cat([features, conv_out], dim=-1)

        # Sampler output
        if self.use_sampler:
            sampler_out = self.sampler(features)
            features = torch.cat([features, sampler_out], dim=-1)

        # Quantum layer placeholder – currently just a linear transformation
        if self.use_quantum:
            # In a real hybrid setting this would be replaced by a quantum
            # expectation value.  For demonstration we use a simple linear layer.
            quantum_feat = torch.sin(features.sum(dim=-1, keepdim=True))
            features = torch.cat([features, quantum_feat], dim=-1)

        return self.estimator(features)
