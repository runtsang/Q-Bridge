"""Hybrid 2‑D convolutional layer combining a classical kernel and a quantum feature extractor.

The class implements the same `run` API as the original seed, but now supports
- learnable convolution weights (via `nn.Conv2d`);
- optional batch‑normalization and bias;
- a `forward` method that can be used inside a larger PyTorch model.

The quantum module can be used as a drop‑in replacement for the original
`Conv` filter in any pipeline that expects a 2‑D filter returning a scalar.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional

try:
    from.quantum import QuantumConvolution
except Exception:  # pragma: no cover
    QuantumConvolution = None  # type: ignore


class ConvGen(nn.Module):
    """Hybrid convolutional filter with optional quantum back‑end.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel.
    threshold : float, default 0.0
        Threshold used by the classical convolution.
    q_params : dict | None, default None
        Parameters for the quantum circuit (`shots`, `backend`).  If `None`,
        the layer operates purely classically.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        *,
        q_params: Optional[dict] = None
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Classical part
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
            padding="same",
        )
        self.bn = nn.BatchNorm2d(1)

        # Optional quantum component
        self.qcircuit: Optional[QuantumConvolution] = None
        if q_params and QuantumConvolution is not None:
            self.qcircuit = QuantumConvolution(
                n_qubits=kernel_size ** 2,
                shots=q_params.get("shots", 100),
                backend=q_params.get("backend"),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning a single‑channel feature map.

        The output shape is ``(batch, 1, H, W)`` matching the input shape.
        """
        # Classical convolution
        out = self.conv(x)
        out = torch.sigmoid(out - self.threshold)
        out = self.bn(out)

        if self.qcircuit:  # pragma: no cover
            # Extract patches and run quantum circuit on each
            batch, _, h, w = out.shape
            patches = out.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            # patches shape: (batch, 1, h, w, kernel_size, kernel_size)
            patches = patches.squeeze(1)
            patches = patches.reshape(batch * h * w, self.kernel_size ** 2)
            # Get quantum feature map
            quantum_features = self.qcircuit.get_feature_map(patches)
            # Reshape back to spatial map
            quantum_features = quantum_features.reshape(batch, 1, h, w)
            out = quantum_features

        return out

    def run(self, data: torch.Tensor) -> float:
        """Compatibility wrapper that mimics the original seed's API.

        Accepts a 2‑D input array and returns the mean of the output feature map.
        """
        with torch.no_grad():
            out = self.forward(data.unsqueeze(0).unsqueeze(0))
            return out.mean().item()


__all__ = ["ConvGen"]
