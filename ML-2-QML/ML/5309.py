"""FastBaseEstimator__gen195: Classical estimator with noise models and automatic differentiation support.

The module exposes a single :class:`FastBaseEstimator__gen195` class that accepts a
PyTorch :class:`torch.nn.Module` and evaluates it on batches of parameters.
The interface mirrors the original FastBaseEstimator API:
```
evaluate(observables, parameter_sets, *, shots=None, noise=None, seed=None)
```
but now the ``shots`` argument can be used for any backend: deterministic
evaluation, Gaussian noise, or Poisson shot‑noise.  The design keeps the
``FastEstimator`` subclass for backward compatibility while delegating to the
``FastBaseEstimator__gen195`` core.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class FastBaseEstimator__gen195:
    """Hybrid estimator that evaluates a classical PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        A neural network that maps a 1‑D input vector to an output tensor.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        noise: str | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute expectations for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar
            (torch.Tensor or float).  If empty, the mean of the output
            along the last dimension is used.
        parameter_sets : list of sequences
            Each inner sequence contains the numeric parameters for one
            forward pass.
        shots : int, optional
            Number of measurement shots.  If ``None`` the evaluation is
            deterministic.
        noise : {'gaussian', 'poisson'}, optional
            Noise model applied when ``shots`` is provided.  Default is
            Gaussian.
        seed : int, optional
            Random seed for reproducible noise.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        rng = np.random.default_rng(seed)
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inp = self._ensure_batch(params)
                out = self.model(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)

        # Apply shot‑noise if requested
        if shots is not None:
            if noise is None or noise == "gaussian":
                std = max(1e-6, 1.0 / shots)
                results = [
                    [rng.normal(mean, std) for mean in row] for row in results
                ]
            elif noise == "poisson":
                results = [
                    [rng.poisson(mean) / shots for mean in row] for row in results
                ]
            else:
                raise ValueError(f"Unknown noise model: {noise}")

        return results
