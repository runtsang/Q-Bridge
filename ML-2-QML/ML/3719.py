"""Hybrid estimator combining classical PyTorch models with optional sampling.

The class accepts a PyTorch ``nn.Module`` or a callable returning class
probabilities. It evaluates a batch of parameter sets, optionally adding
Gaussian shot noise, and can perform sampling using a softmax output or a
custom sampler.

Example usage::

    import torch
    from torch import nn
    from FastHybridEstimator import FastHybridEstimator

    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
    estimator = FastHybridEstimator(model)
    obs = [lambda out: out.mean(dim=-1)]  # default observable
    params = [[0.1, 0.2], [0.3, 0.4]]
    results = estimator.evaluate(obs, params, shots=1000, seed=42)

    # sampling
    samples = estimator.sample(params[0], n=10, seed=42)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
SamplerCallable = Callable[[torch.Tensor], torch.Tensor]  # returns probabilities


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D batched tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Estimator that works with either a PyTorch model or a sampling callable.

    Parameters
    ----------
    model_or_sampler : nn.Module | SamplerCallable
        If a ``nn.Module`` is provided the forward pass is interpreted as a
        regression output.  If a callable is provided it must return
        class probabilities and the estimator will expose a ``sample`` method.
    """

    def __init__(self, model_or_sampler: Union[nn.Module, SamplerCallable]) -> None:
        if isinstance(model_or_sampler, nn.Module):
            self._model = model_or_sampler
            self._sampler: Optional[SamplerCallable] = None
        else:
            self._model = None
            self._sampler = model_or_sampler

        self._is_model = self._model is not None

    # ------------------ evaluation ------------------ #
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Batch‑evaluate observables for a list of parameter sets.

        Parameters
        ----------
        observables
            A collection of callables that accept the model output
            (or sampler output) and return a scalar.
        parameter_sets
            A sequence of 1‑D sequences of parameters.
        shots
            If provided, Gaussian shot noise is added with variance 1/shots.
        seed
            Random seed for the Gaussian noise.

        Returns
        -------
        List[List[float]]
            Outer list over parameter sets, inner list over observables.
        """
        if self._is_model:
            return self._evaluate_model(observables, parameter_sets, shots, seed)
        else:
            return self._evaluate_sampler(observables, parameter_sets, shots, seed)

    def _evaluate_model(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        shots: Optional[int],
        seed: Optional[int],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self._model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self._model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results

    def _evaluate_sampler(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        shots: Optional[int],
        seed: Optional[int],
    ) -> List[List[float]]:
        # For a sampler we treat the output as probability distribution.
        # Observables are expectation values of custom functions over the distribution.
        results: List[List[float]] = []
        for params in parameter_sets:
            probs = self._sampler(torch.as_tensor(params, dtype=torch.float32))
            probs = probs.detach().cpu().numpy()
            row: List[float] = []
            for observable in observables:
                val = observable(torch.tensor(probs))
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy
        return results

    # ------------------ sampling ------------------ #
    def sample(
        self,
        parameter_set: Sequence[float],
        n: int = 1,
        seed: Optional[int] = None,
    ) -> List[int]:
        """Return ``n`` samples from the probability distribution.

        For a pure model a softmax layer is required; the estimator will
        raise if the model does not expose probabilities.
        For a sampler callable the probability distribution must be returned
        directly.

        Parameters
        ----------
        parameter_set
            Sequence of parameters for a single evaluation.
        n
            Number of samples to draw.
        seed
            Random seed for reproducibility.

        Returns
        -------
        List[int]
            Sample indices.
        """
        if self._is_model and self._sampler is None:
            # try to infer a probability distribution from the model
            with torch.no_grad():
                out = self._model(torch.as_tensor(parameter_set, dtype=torch.float32))
            probs = F.softmax(out, dim=-1)
        else:
            probs = self._sampler(torch.as_tensor(parameter_set, dtype=torch.float32))

        probs = probs.detach().cpu().numpy()
        rng = np.random.default_rng(seed)
        return rng.choice(len(probs), size=n, p=probs).tolist()

__all__ = ["FastHybridEstimator"]
