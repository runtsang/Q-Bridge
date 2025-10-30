"""
ConvGen171 – Classical hybrid filter with quantum‑inspired parameterization.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
from typing import Iterable


def _random_unitary(dim: int) -> Tensor:
    """
    Generate a random unitary matrix using QR decomposition of a complex Gaussian matrix.
    The matrix is returned as a real tensor of shape (dim, dim) that will be used as a
    linear transformation on flattened patches.
    """
    rng = np.random.default_rng()
    z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(z)
    return torch.from_numpy(np.real(q)).float()


class ConvGen171(nn.Module):
    """
    A hybrid convolutional filter that combines a learnable convolution with a
    quantum‑inspired linear transformation.  The filter accepts a 2‑D patch,
    applies a conventional learnable Conv2d, then passes the flattened output
    through a random unitary matrix to emulate a quantum kernel.  The final
    activation is a sigmoid of the mean value, optionally thresholded.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 1,
        out_channels: int = 4,
        threshold: float = 0.0,
        use_qconv: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold
        self.use_qconv = use_qconv

        # Classical convolution
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )

        # Quantum‑inspired linear transformation
        dim = out_channels * kernel_size * kernel_size
        self.register_buffer("unitary", _random_unitary(dim))

    def _qconv_forward(self, x: Tensor) -> Tensor:
        """
        Apply the random unitary to the flattened convolution output and
        compute the expectation value as a scalar.
        """
        bsz, c, h, w = x.shape
        flat = x.view(bsz, -1)  # shape: (bsz, dim)
        transformed = flat @ self.unitary  # linear transformation
        # Expectation: mean of absolute values (mimics measurement statistics)
        return transformed.abs().mean(dim=1, keepdim=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the hybrid filter.

        Args:
            x: Tensor of shape (batch, 1, H, W)

        Returns:
            Tensor of shape (batch, 1) containing the activation per sample.
        """
        conv_out = self.conv(x)  # (batch, out_channels, H', W')
        if self.use_qconv:
            q_out = self._qconv_forward(conv_out)  # (batch, 1)
        else:
            # Classical alternative: flatten and apply sigmoid
            flat = conv_out.view(conv_out.size(0), -1)
            q_out = torch.sigmoid(flat - self.threshold).mean(dim=1, keepdim=True)
        return q_out

    def run(self, data: np.ndarray) -> float:
        """
        Convenience method to process a single 2‑D patch and return a scalar.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(tensor)
        return out.item()


class ConvGen171Classifier(nn.Module):
    """
    Simple classifier that stacks the hybrid filter with a fully‑connected head.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 1,
        out_channels: int = 4,
        num_classes: int = 10,
        use_qconv: bool = False,
    ) -> None:
        super().__init__()
        self.feature_extractor = ConvGen171(
            kernel_size=kernel_size,
            stride=stride,
            out_channels=out_channels,
            use_qconv=use_qconv,
        )
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.feature_extractor(x)
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["ConvGen171", "ConvGen171Classifier"]
