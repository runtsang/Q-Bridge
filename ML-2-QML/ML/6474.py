"""Hybrid estimator that evaluates classical neural networks with batched support and optional Gaussian noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class FastHybridEstimator:
    """Evaluate a PyTorch model for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to be evaluated.
    device : str | None, optional
        Device on which to run the model. If None, ``torch.device('cpu')`` is used.
    """

    def __init__(self, model: nn.Module, device: str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output tensor and returns a scalar
            (or a tensor that can be reduced to a scalar).
        parameter_sets : sequence of parameter sequences
            Each inner sequence contains the input values for a single forward pass.
        shots : int | None, optional
            If provided, Gaussian shot noise with standard deviation ``1/shots`` is added
            to the deterministic result. ``None`` means deterministic evaluation.
        seed : int | None, optional
            Random seed for reproducibility when ``shots`` is set.

        Returns
        -------
        List[List[float]]
            A list of rows, one per parameter set, each containing the observables.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]

        # Convert all parameter sets to a single batch tensor
        batch = torch.tensor(parameter_sets, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)
            results: List[List[float]] = []
            for out in outputs:
                row: List[float] = []
                for observable in observables:
                    val = observable(out)
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
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

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss between predictions and targets using a user‑supplied loss function."""
        return loss_fn(predictions, targets)

__all__ = ["FastHybridEstimator"]
