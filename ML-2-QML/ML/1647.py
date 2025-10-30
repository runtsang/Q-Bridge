"""Hybrid classical convolution module with optional quantum‑inspired post‑processing.

Conv() returns an instance of :class:`ConvGen`. The class inherits from
``torch.nn.Module`` and provides a ``run()`` method compatible with the
original seed. When ``use_quantum=True`` a tiny variational network is
applied after the convolution to mimic quantum feature extraction, but
the whole implementation remains classical and fully differentiable.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Conv"]


class ConvGen(nn.Module):
    """Convolutional filter with optional quantum‑inspired layer.

    Parameters
    ----------
    kernel_size : int
        Size of the square filter.
    threshold : float
        Value subtracted from the convolution logits before sigmoid.
    use_quantum : bool
        If True, a tiny variational network is applied to the convolution
        output to mimic quantum feature extraction.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_quantum: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum

        # Base 2‑D convolution mimicking the original filter
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )

        # Quantum‑inspired post‑processing if requested
        if self.use_quantum:
            # A small feed‑forward network that learns a non‑linear
            # mapping from the flattened convolution feature to a
            # probability.
            self.quantum_layer = nn.Sequential(
                nn.Linear(kernel_size * kernel_size, kernel_size * kernel_size),
                nn.Sigmoid(),
            )

    def run(self, data: torch.Tensor | list | tuple | np.ndarray) -> float:
        """Apply convolution (and optional quantum layer) to *data*.

        Parameters
        ----------
        data
            2‑D array of shape (kernel_size, kernel_size) containing
            integer pixel values.

        Returns
        -------
        float
            Mean activation after the sigmoid (or after the quantum‑
            inspired layer if enabled).
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)

        if self.use_quantum:
            flat = activations.view(-1)
            probs = self.quantum_layer(flat)
            return probs.mean().item()
        else:
            return activations.mean().item()


def Conv(*args, **kwargs) -> ConvGen:
    """Drop‑in replacement for the original Conv factory.

    Any arguments are forwarded to :class:`ConvGen`.  The function
    simply returns an instantiated object so existing code that calls
    ``Conv()`` continues to run unchanged.
    """
    return ConvGen(*args, **kwargs)
