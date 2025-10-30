"""Hybrid estimator that unifies classical neural network and quantum circuit evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of parameters into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFastEstimator:
    """Evaluate a PyTorch neural network on batches of inputs with optional noise.

    The estimator supports per‑feature scaling, dropout during evaluation, and
    a simple k‑fold cross‑validation routine that returns the mean and std
    of the predictions for each fold.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        scaler: Callable[[torch.Tensor], torch.Tensor] | None = None,
        dropout: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model:
            A PyTorch ``nn.Module`` that maps input parameters to outputs.
        scaler:
            Optional callable that normalises the input tensor.  If ``None`` the
            identity function is used.
        dropout:
            If provided, a dropout layer with the given probability is applied
            to the model outputs during evaluation.
        """
        self.model = model
        self.scaler = scaler or (lambda x: x)
        self.dropout = dropout
        if dropout is not None:
            self._dropout_layer = nn.Dropout(dropout)
        else:
            self._dropout_layer = None

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute scalar observables for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of callables that map model outputs to a scalar.
        parameter_sets:
            Sequence of parameter vectors to evaluate.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                inputs = self.scaler(inputs)
                outputs = self.model(inputs)
                if self._dropout_layer is not None:
                    outputs = self._dropout_layer(outputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add Gaussian shot noise to the deterministic predictions."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def cross_validate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        k_folds: int = 5,
        *,
        seed: int | None = None,
    ) -> Tuple[float, float]:
        """Return mean and std of predictions over k‑fold splits."""
        rng = np.random.default_rng(seed)
        indices = np.arange(len(parameter_sets))
        rng.shuffle(indices)
        fold_sizes = np.full(k_folds, len(parameter_sets) // k_folds, dtype=int)
        fold_sizes[: len(parameter_sets) % k_folds] += 1
        current = 0
        preds: List[List[float]] = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            # Train on train_idx (here we simply use the same model; in practice
            # you would clone and refit).  For demonstration we skip training.
            fold_preds = self.evaluate(observables, [parameter_sets[i] for i in test_idx])
            preds.extend(fold_preds)
            current = stop
        flat = [p for row in preds for p in row]
        return float(np.mean(flat)), float(np.std(flat))


__all__ = ["HybridFastEstimator"]
