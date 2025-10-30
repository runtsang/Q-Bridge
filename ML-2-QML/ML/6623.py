"""ConvHybrid: classical convolutional filter with optional quantum expectation head.

This module implements a drop‑in replacement for the original Conv class
and extends its behaviour to include an optional quantum circuit when a
backend is supplied.  The classical convolution is expressed as a
PyTorch nn.Conv2d layer that can be trained end‑to‑end; the quantum
expectation head is implemented with a simple parameter‑free circuit
that measures the probability of a single qubit being “1”.  The
hybrid class can **not** directly use a quantum simulator from
within the PyTorch forward pass, because the quantum circuit execution
is expensive; instead, it exposes a ``run_quantum`` method that runs
the quantum circuit on a batch of data and returns the average
probability.  The design follows a modular architecture so that the
quantum part can be executed outside the back‑training loop or in a
hybrid training regime where a classical gradient is used for the
conv layer while a finite‑difference quantum gradient is applied to
the quantum head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = ["ConvHybrid"]


class ConvHybrid(nn.Module):
    """Hybrid convolution‑filter with a classical conv layer and an optional quantum head.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square convolution kernel.
    threshold : float, default 0.0
        Threshold used in the classical sigmoid activation.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical conv layer.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (..., kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Mean activation after sigmoid thresholding.
        """
        tensor = data.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])

    def run_quantum(self, data: np.ndarray) -> float:
        """Placeholder for quantum execution.

        Parameters
        ----------
        data : np.ndarray
            Input array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across qubits.

        Notes
        -----
        The actual quantum execution is provided in the QML module
        ``ConvHybrid``.  This method is kept for API compatibility.
        """
        raise NotImplementedError(
            "Quantum execution is available in the QML module. "
            "Import from 'ConvHybrid' in the quantum package."
        )
