"""Hybrid convolutional filter that combines a classical convolution with a parameterised linear
block and exposes a FastBaseEstimator‑style evaluation interface.

The model is a drop‑in replacement for the original Conv filter while allowing batch
evaluation of arbitrary observables and optional Gaussian shot noise to emulate a
quantum backend.

The implementation is purely classical and relies only on PyTorch.
"""

from __future__ import annotations

import torch
from torch import nn
from collections.abc import Iterable, Sequence
import numpy as np
from typing import Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridConvEstimator(nn.Module):
    """Classical hybrid filter that mimics the behaviour of the quantum quanvolution
    while providing a FastBaseEstimator‑style evaluation interface.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter.
    threshold : float, default 0.0
        Threshold for the sigmoid activation.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.fc = nn.Linear(kernel_size ** 2, 1, bias=False)

    def forward(self, data: torch.Tensor, params: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the convolution, sigmoid thresholding and the parametric linear block.

        Parameters
        ----------
        data : torch.Tensor
            2‑D tensor of shape (kernel_size, kernel_size).
        params : torch.Tensor, optional
            1‑D tensor of length kernel_size**2 containing per‑feature scaling factors.
            If ``None`` the layer is applied with its current weights.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the final prediction.
        """
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        features = activations.view(-1)
        if params is not None:
            features = features * params
        out = self.fc(features)
        return out.squeeze()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[float]]:
        """Evaluate a list of observables for a batch of parameter sets.

        Parameters
        ----------
        observables : Iterable[ScalarObservable], optional
            Callables that map the model output to a scalar.
        parameter_sets : Sequence[Sequence[float]], optional
            List of parameter vectors applied to the ``fc`` layer.
        shots : int, optional
            If provided, Gaussian noise with variance ``1/shots`` is added to each
            expectation value to emulate shot noise.
        seed : int, optional
            Random seed for the shot noise generator.

        Returns
        -------
        list[list[float]]
            Nested list where each inner list contains the values of all observables
            for a single parameter set.
        """
        if observables is None:
            observables = [lambda x: x]
        if parameter_sets is None:
            parameter_sets = [()]

        results: list[list[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                param_tensor = _ensure_batch(params).squeeze()
                dummy_input = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
                out = self.forward(dummy_input, param_tensor)
                row: list[float] = []
                for observable in observables:
                    value = observable(out)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: list[list[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy
        return results


__all__ = ["HybridConvEstimator"]
