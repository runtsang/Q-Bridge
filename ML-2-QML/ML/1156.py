"""Hybrid classical convolutional feature extractor for ConvGen092."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ConvGen092(nn.Module):
    """
    Drop‑in replacement for the original Conv filter.
    Implements a two‑stage pipeline:
    1. A shallow CNN that supports batch and multi‑channel input.
    2. The output of each convolution window is fed into a
       variational quantum circuit (passed in as a callable).
    The quantum filter must accept a 1‑D numpy array of length
    ``kernel_size**2 * out_channels`` and return a float.
    """

    def __init__(
        self,
        *,  # keyword‑only arguments
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 2,
        quantum_filter: callable | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        # ---------- Classical CNN ----------
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # ---------- Quantum filter ----------
        # If no filter is supplied, fall back to a deterministic
        # classical proxy that mimics the original behaviour.
        if quantum_filter is None:
            def default_filter(x: np.ndarray) -> float:
                """Deterministic proxy for the quantum filter."""
                return float(np.mean(np.tanh(x)))
            self.quantum_filter = default_filter
        else:
            self.quantum_filter = quantum_filter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the CNN followed by the quantum filter.
        The quantum filter is applied to each convolution window.
        """
        # Classical feature extraction
        feat = self.cnn(x)  # shape: (B, C, H, W)

        B, C, H, W = feat.shape
        k = self.kernel_size

        # Prepare output list
        out_vals = []

        for b in range(B):
            for i in range(H - k + 1):
                for j in range(W - k + 1):
                    # Extract the window across all channels
                    window = feat[b, :, i : i + k, j : j + k]
                    # Flatten to 1‑D vector
                    vec = window.detach().cpu().numpy().reshape(-1)
                    # Apply quantum filter
                    q_val = self.quantum_filter(vec)
                    out_vals.append(q_val)

        # Return as a 1‑D tensor (batch × spatial windows)
        return torch.tensor(out_vals, device=x.device)

def Conv() -> ConvGen092:
    """Convenience factory that returns a ConvGen092 instance."""
    return ConvGen092()
