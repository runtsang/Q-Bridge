from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """
    Hybrid estimator for classical neural nets and quantum circuits.

    Parameters
    ----------
    model
        Either a torch.nn.Module (for classical inference) or any object
        providing a ``run`` method that returns a torch.Tensor.

    The ``evaluate`` method accepts an iterable of callables (acting as
    observables) and a sequence of parameter vectors.  If ``shots`` is
    provided, Gaussian shot noise with variance 1/shots is added to each
    output.  The estimator is API‑compatible with the original
    FastBaseEstimator, making it drop‑in for existing pipelines.
    """

    def __init__(self, model: nn.Module | object) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate each observable for every parameter set.

        Parameters
        ----------
        observables
            Callables taking a tensor and returning a scalar.
        parameter_sets
            Sequence of parameter vectors.
        shots
            If provided, adds Gaussian shot noise with variance 1/shots.
        seed
            Seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            Outer list per parameter set, inner list per observable.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        # Determine if the model is a PyTorch module
        is_torch = isinstance(self.model, nn.Module)

        for params in parameter_sets:
            if is_torch:
                self.model.eval()
                with torch.no_grad():
                    inputs = _ensure_batch(params)
                    outputs = self.model(inputs)
            else:
                # Assume model has a run method returning a torch‑compatible array
                out = self.model.run(params)
                outputs = torch.as_tensor(out, dtype=torch.float32)

            row: List[float] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = float(value.mean().cpu())
                else:
                    scalar = float(value)
                row.append(scalar)

            if shots is not None:
                rng = np.random.default_rng(seed)
                row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            results.append(row)

        return results


__all__ = ["HybridEstimator"]
