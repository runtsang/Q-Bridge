import torch
from torch import nn
import numpy as np
from typing import Iterable, Tuple

class HybridFCL(nn.Module):
    """Classical fully‑connected layer with fraud‑detection style scaling.

    Combines a linear + tanh block with optional clipping and amplitude
    scaling/shift buffers that mirror the displacement and phase‑shift
    operations used in the quantum photonic circuit.  The implementation
    is fully PyTorch‑based, making it usable in any classical training
    pipeline.
    """

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 1,
        clip: bool = True,
        clip_bounds: Tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.activation = nn.Tanh()
        # Buffers that emulate displacement and phase‑shift from the quantum side
        self.register_buffer("scale", torch.ones(out_features, dtype=torch.float32))
        self.register_buffer("shift", torch.zeros(out_features, dtype=torch.float32))
        self.clip = clip
        self.clip_bounds = clip_bounds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional clipping of weights/biases."""
        if self.clip:
            with torch.no_grad():
                self.linear.weight.copy_(self.linear.weight.clamp(*self.clip_bounds))
                self.linear.bias.copy_(self.linear.bias.clamp(*self.clip_bounds))
        y = self.activation(self.linear(x))
        return y * self.scale + self.shift

    def set_scale_shift(self, scale: Iterable[float], shift: Iterable[float]) -> None:
        """Set the amplitude scale and shift buffers."""
        self.scale.copy_(torch.tensor(scale, dtype=torch.float32))
        self.shift.copy_(torch.tensor(shift, dtype=torch.float32))

__all__ = ["HybridFCL"]
