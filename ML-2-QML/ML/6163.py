"""Hybrid estimator for classical neural network models.

This module implements :class:`FastHybridEstimator` that wraps a PyTorch
``nn.Module``.  It inherits the deterministic evaluation logic from the
original FastBaseEstimator, but also adds optional Gaussian shot noise and
a lightweight convolutional pre‑processing step.  The convolution filter
is implemented by the :func:`Conv` factory from the classic seed.

The estimator is intentionally independent from any quantum backend,
so it can be used in pure classical pipelines or as a drop‑in
replacement for the quantum version when a classical model is desired.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Sequence, Union

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of parameters to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# --------------------------------------------------------------------------- #
# Classical convolution filter (drop‑in replacement for quanvolution)
# --------------------------------------------------------------------------- #

try:
    # Import the lightweight Conv factory from the classic seed.
    from.Conv import Conv as ClassicalConv  # type: ignore
except Exception:  # pragma: no cover
    # Fallback: simple PyTorch implementation if the import fails.
    import torch.nn as nn

    def ClassicalConv(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
        class ConvFilter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.kernel_size = kernel_size
                self.threshold = threshold
                self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

            def run(self, data) -> float:
                tensor = torch.as_tensor(data, dtype=torch.float32)
                tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
                logits = self.conv(tensor)
                activations = torch.sigmoid(logits - self.threshold)
                return activations.mean().item()

        return ConvFilter()


# --------------------------------------------------------------------------- #
# Core estimator
# --------------------------------------------------------------------------- #

class FastHybridEstimator:
    """
    Evaluate a PyTorch model for a batch of parameter sets and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.  It must accept a 2‑D tensor of shape
        ``(batch, features)`` and return a 2‑D tensor of outputs.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        conv_filter: nn.Module | None = None,
    ) -> List[List[float]]:
        """
        Compute observables for each set of parameters.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar.
            If empty, the mean of the outputs is used.
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameters for one evaluation.
        shots : int, optional
            If provided, Gaussian noise with variance ``1/shots`` is added
            to each result, emulating shot noise.
        seed : int, optional
            Seed for the noise generator.
        conv_filter : nn.Module, optional
            A callable with a ``run`` method that accepts raw input data
            and returns a scalar.  It is applied *before* the model
            evaluation; useful for classical quanvolution.
        """
        # Apply convolutional pre‑processing if requested.
        if conv_filter is not None:
            # The filter is expected to output a scalar; we broadcast it
            # to match the model input shape.
            conv_outputs = [conv_filter.run(np.array(p)) for p in parameter_sets]
            # Convert to a 2‑D tensor for the model.
            inputs = torch.tensor(conv_outputs, dtype=torch.float32).unsqueeze(-1)
        else:
            inputs = _ensure_batch([p[0] for p in parameter_sets])  # simple 1‑D case

        # Default observable: mean of outputs
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]

        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # Broadcast parameters to match the input shape
                if conv_filter is None:
                    inputs = _ensure_batch(params)
                else:
                    # Convolution already applied
                    pass

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

        # Add shot noise if requested
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


__all__ = ["FastHybridEstimator"]
