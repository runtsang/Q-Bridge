"""Hybrid estimator supporting PyTorch models, optional autoencoders, and shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model over batches of inputs and optional observables.

    Parameters
    ----------
    model:
        A `torch.nn.Module` or a factory callable returning such a module.
    autoencoder:
        Optional autoencoder (`nn.Module`) that preprocesses the inputs.
    seed:
        Seed for the internal noise generator.  Useful when the same estimator
        is reused with different shot configurations.
    """

    def __init__(
        self,
        model: Union[nn.Module, Callable[[], nn.Module]],
        *,
        autoencoder: nn.Module | None = None,
        seed: int | None = None,
    ) -> None:
        if callable(model) and not isinstance(model, nn.Module):
            self.model = model()
        else:
            self.model = model
        self.autoencoder = autoencoder
        self.rng = np.random.default_rng(seed)

    def _forward(self, params: torch.Tensor) -> torch.Tensor:
        if self.autoencoder is not None:
            params = self.autoencoder(params)
        return self.model(params)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a matrix of observable values for each parameter set.

        If ``shots`` is provided, Gaussian noise with variance ``1/shots`` is added
        to each observable value, mimicking measurement shot noise.
        """
        rng = self.rng if shots is None else np.random.default_rng(seed)

        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self._forward(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                if shots is not None:
                    row = [float(rng.normal(v, max(1e-6, 1 / shots))) for v in row]
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Convenience subclass that always injects shot noise."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        return super().evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["FastBaseEstimator", "FastEstimator"]
