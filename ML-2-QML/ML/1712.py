"""Enhanced neural‑network estimator with GPU support and flexible noise models.

The class keeps the spirit of the original FastBaseEstimator: it evaluates a
PyTorch module for a list of parameter sets and a collection of scalar
observables.  The implementation is intentionally lightweight so that it can
be embedded into larger experiment pipelines, but it adds:

* optional device selection (CPU/GPU)
* automatic torch.compile support for PyTorch ≥ 2.0
* vectorised evaluation of batched parameters
* pluggable shot‑noise models (default Gaussian, custom callable)

The API remains backward compatible – the constructor accepts the same
``model`` argument and the ``evaluate`` signature is unchanged, except for
additional keyword arguments controlling the noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor of shape ``(1, N)`` for a single parameter set."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for multiple parameter sets and observables.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` whose ``forward`` method accepts a tensor of shape
        ``(batch, dim)`` and returns a tensor of shape ``(batch, out_dim)``.
    device:
        Target device, e.g. ``"cpu"`` or ``"cuda"``.  The model is moved to the
        device automatically.
    compile:
        If ``True`` and PyTorch >= 2.0, the model will be compiled with
        ``torch.compile`` for slightly faster inference.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        compile: bool = False,
    ) -> None:
        self.model = model.to(device)
        if compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute the expectation value of each observable for every set of
        parameters.

        Parameters
        ----------
        observables:
            A sequence of callables that map the model output tensor to a
            scalar (float or tensor).  If empty the average of the last
            dimension is returned by default.
        parameter_sets:
            Iterable of parameter vectors.  Each vector is converted to a
            tensor and supplied to the model.

        Returns
        -------
        List[List[float]]:
            Nested list where the outer list corresponds to parameter sets and
            the inner list to observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            # Vectorised evaluation if possible
            batch = torch.cat([_ensure_batch(p) for p in parameter_sets], dim=0)
            batch = batch.to(self.device)
            outputs = self.model(batch)

            for idx, params in enumerate(parameter_sets):
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs[idx : idx + 1])
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Add configurable shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        noise_fn: Optional[Callable[[float], float]] = None,
    ) -> List[List[float]]:
        """Return noisy estimates of the observables.

        Parameters
        ----------
        shots:
            Number of measurement shots.  If ``None`` no noise is applied.
        seed:
            Random seed for reproducibility.
        noise_fn:
            Callable that takes the mean value and returns a noisy float.  If
            ``None`` a Gaussian noise with variance ``1/shots`` is used.

        Returns
        -------
        List[List[float]]:
            Nested list of noisy estimates.
        """
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []

        for row in raw:
            if noise_fn is None:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
            else:
                noisy_row = [noise_fn(mean) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
