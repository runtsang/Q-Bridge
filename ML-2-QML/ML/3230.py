"""Hybrid estimator combining classical PyTorch models and a quanvolution filter.

The estimator accepts any nn.Module and evaluates it on sequences of
parameter vectors.  Optional Gaussian shot noise can be added to emulate
finite sampling.  A helper builds a simple quanvolution classifier
based on a 2×2 convolutional kernel.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _to_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables.

    Parameters
    ----------
    model:
        Any :class:`torch.nn.Module`.  It may be a pure classical network or a
        hybrid model that contains a quantum filter implemented with
        :mod:`torchquantum`.  The estimator will simply forward the inputs
        through the model and apply the supplied observables.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute scalar observables for each parameter set.

        If *shots* is provided, Gaussian noise with variance 1/shots is added
        to each observation to emulate finite‑shot sampling.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _to_batch(params).to(
                    self.model.device if hasattr(self.model, "device") else "cpu"
                )
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)

        return noisy

    @staticmethod
    def build_quanvolution_classifier() -> nn.Module:
        """Return a simple hybrid classifier that uses a 2×2 quanvolution filter."""
        class QuanvolutionFilter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                features = self.conv(x)
                return features.view(x.size(0), -1)

        class QuanvolutionClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.qfilter = QuanvolutionFilter()
                self.linear = nn.Linear(4 * 14 * 14, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                features = self.qfilter(x)
                logits = self.linear(features)
                return nn.functional.log_softmax(logits, dim=-1)

        return QuanvolutionClassifier()
