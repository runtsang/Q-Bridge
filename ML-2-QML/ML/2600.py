"""Hybrid estimator for classical models with optional convolutional preprocessing and shot noise.

This module defines FastBaseEstimator that can evaluate a PyTorch neural network on batches
of parameters. It extends the original lightweight estimator by adding:

* Optional convolutional preprocessing (classical ConvFilter) that can be applied to 2‑D
  inputs before feeding them to the network.
* Automatic GPU support and batch‑tensor conversion.
* Flexible observable interface with default mean‑output observable.
* Gaussian shot‑noise simulation that mirrors a quantum measurement.

The estimator remains fully classical and can be used as a drop‑in replacement for the
original FastBaseEstimator while providing richer preprocessing and noise modelling.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Convolutional preprocessing (classical)
# --------------------------------------------------------------------------- #
def ConvFilter(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
    """Return a PyTorch module that emulates a quantum filter with a 2‑D convolution.

    The implementation is a lightweight wrapper around ``nn.Conv2d`` that
    applies a sigmoid activation and averages the result.  It is inspired
    by the original ``Conv`` seed but exposes a more explicit API and
    type hints.

    Args:
        kernel_size: Size of the square kernel.
        threshold: Bias term applied before the sigmoid.

    Returns:
        nn.Module: A callable module that can be inserted into a
        larger network or used as a stand‑alone preprocessor.
    """
    class _Conv(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=kernel_size, bias=True
            )

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            # Expect data shape (batch, height, width) or (height, width)
            if data.ndim == 2:
                data = data.unsqueeze(0).unsqueeze(0)
            elif data.ndim == 3 and data.shape[0]!= 1:
                # Assume batch dimension present
                data = data.unsqueeze(1)
            logits = self.conv(data)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean(dim=[-2, -1])  # average over spatial dims

    return _Conv()


# --------------------------------------------------------------------------- #
# Main estimator
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model with optional preprocessing and shot noise."""

    def __init__(
        self,
        model: nn.Module,
        *,
        preprocessor: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            model: The neural network to evaluate.
            preprocessor: Optional module that transforms the raw parameters
                before they are fed to *model*.  If ``None`` no preprocessing
                is applied.
            device: Target device; automatically falls back to CPU if ``None``.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute observables for a batch of parameter sets.

        The method first applies the optional *preprocessor*, then runs the
        model in evaluation mode, and finally evaluates each observable.
        If *shots* is provided, Gaussian noise with variance ``1/shots`` is
        added to each mean value to emulate a noisy quantum measurement.

        Args:
            observables: Iterable of callables that map the model output
                to a scalar.  If empty a mean‑output observable is used.
            parameter_sets: Iterable of parameter sequences.
            shots: Number of simulated measurement shots.
            seed: Random seed for reproducibility.

        Returns:
            List of lists containing the observable values for each
            parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                if self.preprocessor is not None:
                    inputs = self.preprocessor(inputs)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = ["FastBaseEstimator", "ConvFilter"]
