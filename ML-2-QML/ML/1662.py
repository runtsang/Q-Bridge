"""FastBaseEstimator with caching and shot noise for PyTorch models."""
import numpy as np
import torch
from torch import nn
from typing import Callable, List, Sequence, Iterable, Dict, Tuple

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Ensure 2D batch tensor for model inference."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """A fast estimator for PyTorch models with optional shot noise and caching.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to evaluate.
    cache : bool, default=False
        Whether to cache intermediate model outputs for repeated parameter sets.
    """

    def __init__(self, model: nn.Module, *, cache: bool = False) -> None:
        self.model = model
        self._cache_enabled = cache
        self._cache: Dict[Tuple[float,...], torch.Tensor] = {}

    def _forward(self, params: Sequence[float]) -> torch.Tensor:
        key = tuple(params)
        if self._cache_enabled and key in self._cache:
            return self._cache[key]
        self.model.eval()
        with torch.no_grad():
            inputs = _ensure_batch(params)
            outputs = self.model(inputs)
        if self._cache_enabled:
            self._cache[key] = outputs
        return outputs

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model on a list of parameter sets.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that map a model output tensor to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter sequences.
        shots : int, optional
            If set, add Gaussian noise with standard deviation 1/sqrt(shots) to each mean.
        seed : int, optional
            Random seed for reproducibility of shot noise.

        Returns
        -------
        List[List[float]]
            Outer list over parameter sets, inner list over observables.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        rng = np.random.default_rng(seed)
        results: List[List[float]] = []
        for params in parameter_sets:
            outputs = self._forward(params)
            row: List[float] = []
            for obs in observables:
                out = obs(outputs)
                scalar = float(out.mean().cpu()) if isinstance(out, torch.Tensor) else float(out)
                row.append(scalar)
            if shots is not None:
                std = max(1e-6, 1 / np.sqrt(shots))
                row = [rng.normal(loc=val, scale=std) for val in row]
            results.append(row)
        return results

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()


__all__ = ["FastBaseEstimator"]
