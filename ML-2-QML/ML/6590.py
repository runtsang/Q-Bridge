"""FastBaseEstimator – a lightweight neural network evaluator with shot noise simulation.

This module extends the original FastBaseEstimator by adding optional
shot‑noise, GPU support, and batched evaluation. It remains fully
classical and uses PyTorch for fast tensor operations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor with shape (batch, 1) for 1‑D input."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for one or many parameter sets.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate. It must accept a batch of inputs
        with shape ``(batch,...)`` and return a tensor of outputs.
    device : str | torch.device, optional
        Device on which to run the model. Defaults to ``'cpu'``.
    """

    def __init__(self, model: nn.Module, device: Union[str, torch.device] = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
            Callables that map the model output tensor to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence contains the parameters for one forward pass.
        shots : int, optional
            If provided, add Gaussian noise with variance ``1/shots`` to each
            observable. This emulates shot noise on quantum devices.
        seed : int, optional
            Random seed for the shot‑noise generator.

        Returns
        -------
        List[List[float]]
            Rows correspond to parameter sets, columns to observables.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        rng = np.random.default_rng(seed)

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)

                if shots is not None:
                    row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                results.append(row)

        return results


class FastEstimator(FastBaseEstimator):
    """Add an explicit ``evaluate_shots`` method for clarity."""

    def evaluate_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Convenience wrapper that enforces a non‑None ``shots`` argument."""
        return super().evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["FastBaseEstimator", "FastEstimator"]
