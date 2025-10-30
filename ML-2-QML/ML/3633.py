import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Classical estimator for deterministic PyTorch models.

    Supports evaluation of a list of scalar observables and optional
    Gaussian noise to emulate finite sampling.  The ``shots`` argument
    controls the noise level, with variance 1/shots.  When ``shots`` is
    ``None`` the estimator is deterministic.
    """

    def __init__(self, model: nn.Module):
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
        """
        Evaluate the model for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and returns either
            a tensor or a scalar.  If empty, the mean of the output is used.
        parameter_sets : sequence of sequences
            Batch of parameters to feed into the model.
        shots : int or None
            If provided, Gaussian noise with std = 1/√shots is added to each
            mean value.
        seed : int or None
            Random seed for noise generation.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().item())
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


__all__ = ["FastBaseEstimator"]
