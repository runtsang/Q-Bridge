"""Adaptive ensemble estimator for PyTorch neural networks."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of parameters to a batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class AdaptiveEstimator:
    """
    Evaluate an ensemble of neural networks on batches of inputs.

    Parameters
    ----------
    models : Sequence[nn.Module]
        One or more PyTorch modules to evaluate.
    dropout : bool, optional
        If ``True``, activates dropout layers during evaluation
        to introduce stochasticity.
    dropout_prob : float, optional
        Dropout probability used when ``dropout=True``.  Defaults to 0.5.
    device : str, optional
        Device to run the models on.  Defaults to ``"cpu"``.
    """

    def __init__(
        self,
        models: Sequence[nn.Module],
        *,
        dropout: bool = False,
        dropout_prob: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.models = [m.to(device) for m in models]
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.device = device

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute mean observable values over the model ensemble.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that map a model output tensor to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors to feed into the models.
        shots : int | None, optional
            If provided, Gaussian shot noise is added with variance
            ``1/shots`` to each mean value.
        seed : int | None, optional
            Random seed for shot noise generation.

        Returns
        -------
        List[List[float]]
            Nested list where each inner list contains the mean observable
            values for a single parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        for params in parameter_sets:
            batch = _ensure_batch(params).to(self.device)

            ensemble_outputs: List[List[float]] = []

            for model in self.models:
                model.eval()
                if self.dropout:
                    model.train()
                with torch.no_grad():
                    output = model(batch)

                row: List[float] = []
                for obs in observables:
                    val = obs(output)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().item())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                ensemble_outputs.append(row)

            mean_row = [sum(vals) / len(vals) for vals in zip(*ensemble_outputs)]
            results.append(mean_row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy_results: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy_results.append(noisy_row)
            return noisy_results

        return results


__all__ = ["AdaptiveEstimator"]
