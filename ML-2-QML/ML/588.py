"""Hybrid estimator that combines a lightweight neural network with optional dropout for uncertainty estimation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """Base class for evaluating neural networks on batches of input parameters with optional dropout and shot noise."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        dropout_runs: Optional[int] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            PyTorch model to evaluate.
        device : str, optional
            Target device for evaluation.
        dropout_runs : int, optional
            Number of stochastic forward passes to average when dropout is enabled.
        shots : int, optional
            Number of shot samples to simulate measurement noise.
        seed : int, optional
            Random seed for reproducibility of shot noise.
        """
        self.model = model.to(device)
        self.device = device
        self.dropout_runs = dropout_runs
        self.shots = shots
        self.seed = seed
        self.model.eval()

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform forward pass with optional dropout averaging."""
        if self.dropout_runs is None or self.dropout_runs <= 1:
            return self.model(inputs)
        # Enable dropout during evaluation
        self.model.train()
        outputs = [self.model(inputs) for _ in range(self.dropout_runs)]
        self.model.eval()
        return torch.mean(torch.stack(outputs), dim=0)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None,
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        Evaluate the model for a batch of parameter sets and observables.

        Parameters
        ----------
        observables : Iterable[ScalarObservable], optional
            Functions mapping model outputs to scalar values.
            If None, the mean of the last dimension is used.
        parameter_sets : Sequence[Sequence[float]]
            Sequence of parameter vectors to evaluate.

        Returns
        -------
        np.ndarray
            Array of shape (n_params, n_observables) containing evaluation results.
        """
        observables = list(observables) if observables else [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            outputs = self._forward(inputs)
            row: List[float] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = float(value.mean().cpu())
                else:
                    scalar = float(value)
                row.append(scalar)
            results.append(row)
        results = np.array(results, dtype=float)
        if self.shots is not None:
            rng = np.random.default_rng(self.seed)
            noise = rng.normal(0, 1 / np.sqrt(self.shots), size=results.shape)
            results = results + noise
        return results


__all__ = ["HybridEstimator"]
