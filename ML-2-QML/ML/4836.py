"""HybridEstimator – classical estimator with optional Conv feature extraction and Gaussian noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Ensure input is a 2‑D batch of floats."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """
    Lightweight PyTorch estimator that can wrap any nn.Module.

    Parameters
    ----------
    model:
        A torch.nn.Module that consumes the output of an optional ConvFilter.
    conv:
        Optional torch.nn.Module used as a feature extractor.  It must
        accept a 2‑D tensor and return a compatible tensor for ``model``.
    """

    def __init__(self, model: nn.Module, conv: nn.Module | None = None) -> None:
        self.model = model
        self.conv = conv

    def _forward(self, params: Sequence[float]) -> torch.Tensor:
        tensor = _ensure_batch(params)
        if self.conv is not None:
            tensor = self.conv(tensor)
        return self.model(tensor)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model for each set of parameters and return a list of
        observable values.  If ``shots`` is supplied, Gaussian shot noise
        is added to model outputs to mimic measurement uncertainty.

        Parameters
        ----------
        observables:
            Callables that map a model output to a scalar value.
        parameter_sets:
            Sequence of numeric parameter lists fed to the model.
        shots:
            Optional shot count for adding noise.
        seed:
            Random seed for reproducibility of noise.
        """
        # Default observable that averages the output
        obs_list = list(observables) or [lambda out: out.mean(dim=-1)]

        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                outputs = self._forward(params)
                row: List[float] = []
                for obs in obs_list:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridEstimator"]
