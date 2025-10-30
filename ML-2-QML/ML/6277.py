"""Hybrid classical‑quantum convolution module.

This module defines a `Conv` class that can operate in three modes:

* **classical** – a single‑channel 2‑D convolution with a learnable bias.
* **quantum** – a placeholder that raises a ``NotImplementedError``; the real
  quantum implementation lives in the QML module.
* **hybrid** – the convolution output is fed into a tiny feed‑forward
  network that approximates the behaviour of a quantum circuit.  This
  makes the module useful for research studies that compare classical
  and quantum pathways without requiring a quantum backend.

The class also supports:

* A learnable activation threshold that can be optimised by gradient descent.
* Batch processing – the forward method accepts a batch of images.
* A ``run`` convenience method that accepts a 2‑D NumPy array and returns a
  scalar activation value, matching the API of the original seed.

"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Union, Literal

class Conv(nn.Module):
    """Hybrid convolutional filter with classical, quantum, and hybrid variants.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    mode : {'classical', 'quantum', 'hybrid'}, default 'classical'
        The execution mode.  ``quantum`` raises a ``NotImplementedError``.
    threshold : float or nn.Parameter, default 0.0
        The activation threshold.  If ``learn_threshold`` is ``True`` the
        threshold is treated as a learnable parameter.
    learn_threshold : bool, default False
        Whether to make the threshold a learnable ``nn.Parameter``.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        mode: Literal["classical", "quantum", "hybrid"] = "classical",
        threshold: Union[float, nn.Parameter] = 0.0,
        learn_threshold: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.mode = mode

        if learn_threshold:
            if isinstance(threshold, float):
                self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
            else:
                self.threshold = threshold
        else:
            self.threshold = threshold

        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        if mode == "hybrid":
            # Small MLP that approximates the quantum layer.
            self.quantum_approx = nn.Sequential(
                nn.Flatten(),
                nn.Linear(kernel_size * kernel_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, H, W)``.  The spatial
            dimensions must be divisible by ``kernel_size`` so that the
            convolution produces a single channel output.

        Returns
        -------
        torch.Tensor
            Activation map after thresholding.  For ``classical`` mode the
            shape is ``(batch, 1, H_out, W_out)``.  For ``hybrid`` mode the
            output is flattened to ``(batch, 1)``.
        """
        conv_out = self.conv(x)

        if self.mode == "classical":
            act = torch.sigmoid(conv_out - self.threshold)
            return act

        if self.mode == "hybrid":
            # Approximate quantum behaviour with a small MLP.
            flat = conv_out.view(conv_out.size(0), -1)
            approx = self.quantum_approx(flat)
            act = torch.sigmoid(approx - self.threshold)
            return act

        raise NotImplementedError("Quantum mode is not available in the classical module.")

    def run(self, data: np.ndarray) -> float:
        """
        Convenience method that mirrors the API of the original seed.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape ``(kernel_size, kernel_size)`` or a batch
            of such arrays with shape ``(batch, kernel_size, kernel_size)``.

        Returns
        -------
        float
            Mean activation over all spatial locations and, for
            batched input, over the batch dimension as well.
        """
        if data.ndim == 2:
            data = data[np.newaxis, np.newaxis, :, :]
        elif data.ndim == 3:
            data = data[:, np.newaxis, :, :]
        else:
            raise ValueError("Input must be 2‑D or 3‑D array.")

        tensor = torch.as_tensor(data, dtype=torch.float32, device=self.conv.weight.device)
        out = self.forward(tensor)
        return out.mean().item()

__all__ = ["Conv"]
