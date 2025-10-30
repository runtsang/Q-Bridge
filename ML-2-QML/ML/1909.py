"""FastBaseEstimator – Classical implementation with advanced inference and metric support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
MetricFunction = Callable[[torch.Tensor, torch.Tensor], float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D list of parameters to a batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """
    Evaluate a PyTorch model on a collection of parameter sets.

    Parameters
    ----------
    model : nn.Module
        Any torch model that accepts a batch of inputs and returns a tensor
        of shape ``(batch, out_dims)``.
    dropout : float | None
        If provided, a Dropout layer with this probability is appended to
        the model during evaluation to simulate measurement noise.
    device : str | torch.device
        Target device for computation (default: cpu).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        dropout: Optional[float] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        if dropout is not None:
            # Wrap the model with a dropout layer for stochastic inference.
            self.model = nn.Sequential(self.model, nn.Dropout(dropout))
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute observables for each set of parameters.

        Returns
        -------
        results : List[ List[float] ]
            Outer list over parameter sets, inner list over observables.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
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
        """
        Add Gaussian shot noise to the deterministic outputs.

        Parameters
        ----------
        shots : int, optional
            Number of shots per observable; if None, no noise is added.
        seed : int, optional
            Random seed for reproducibility.
        """
        deterministic = self.evaluate(observables, parameter_sets)
        if shots is None:
            return deterministic

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in deterministic:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    def compute_metric(
        self,
        metric: MetricFunction,
        targets: Sequence[float],
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[ScalarObservable],
    ) -> float:
        """
        Compute a user‑defined metric between model outputs and target values.

        Parameters
        ----------
        metric : MetricFunction
            Function accepting two tensors ``(pred, target)`` and returning a float.
        targets : Sequence[float]
            Ground‑truth values for each parameter set.
        """
        outputs = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                out = self.model(inputs)
                outputs.append(out.squeeze().cpu())
        pred_tensor = torch.stack(outputs)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32)
        return float(metric(pred_tensor, target_tensor))


__all__ = ["FastBaseEstimator"]
