"""Hybrid convolutional filter combining classical and quantum-inspired kernels.

This module merges concepts from the original Conv.py and Quanvolution.py
to provide a flexible filter that can operate in classical, quantum-inspired,
or hybrid mode. The design is intended to be a drop‑in replacement for the
Conv() factory in the anchor project while exposing additional experimental
capabilities.

Usage:
    from Conv__gen272 import Conv
    filter = Conv(mode='hybrid')
    result = filter.run(image_patch)
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class HybridConvFilter(nn.Module):
    """A convolutional filter that can operate in three modes.

    * ``classical`` – standard 2‑D convolution with learnable weights.
    * ``quantum``   – a random linear transform followed by a sigmoid
      that mimics the probability distribution of a quantum measurement.
    * ``hybrid``    – averages the outputs of the two sub‑modules.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        mode: str = "classical",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.mode = mode.lower()
        if self.mode not in {"classical", "quantum", "hybrid"}:
            raise ValueError(f"Unsupported mode: {mode}")

        # Classical sub‑module
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Quantum‑inspired sub‑module
        # Generate a random weight matrix that will serve as a stand‑in
        # for a variational circuit. The matrix is re‑sampled every forward
        # pass to emulate stochasticity of a quantum kernel.
        self.q_weight_shape = (kernel_size * kernel_size, kernel_size * kernel_size)

    def _classical_forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    def _quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten patch and apply random linear layer
        flat = x.view(1, -1)
        q_weights = torch.randn(self.q_weight_shape, device=x.device)
        transformed = flat @ q_weights
        probs = torch.sigmoid(transformed - self.threshold)
        return probs.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the filter in the selected mode."""
        if self.mode == "classical":
            return self._classical_forward(x)
        if self.mode == "quantum":
            return self._quantum_forward(x)
        # hybrid
        return 0.5 * self._classical_forward(x) + 0.5 * self._quantum_forward(x)

    def run(self, data) -> float:
        """Convenience wrapper that accepts a NumPy array or a torch.Tensor."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        return float(self.forward(tensor).item())

def Conv(mode: str = "classical", kernel_size: int = 2, threshold: float = 0.0):
    """Factory that returns a HybridConvFilter instance.

    Parameters
    ----------
    mode : str
        One of ``'classical'``, ``'quantum'`` or ``'hybrid'``.
    kernel_size : int
        Size of the square patch to process.
    threshold : float
        Activation threshold applied after the sigmoid.
    """
    return HybridConvFilter(kernel_size=kernel_size, threshold=threshold, mode=mode)

__all__ = ["HybridConvFilter", "Conv"]
