"""Hybrid estimator that evaluates a PyTorch QCNN and adds optional shot noise.

This module builds on the classical FastBaseEstimator and the QCNNModel
from the ML reference, but adds a unified interface that supports
multiple observables, shot‑based Gaussian noise, and a flexible
parameter‑binding mechanism.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

# Import the QCNN model factory from the ML seed
from QCNN import QCNNModel, QCNN  # assumes QCNN.py is in the same package

# Type alias for a callable that maps a model output tensor to a scalar
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats into a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Hybrid estimator that can evaluate a PyTorch model (e.g. QCNN) and add Gaussian shot noise.

    The estimator accepts any `torch.nn.Module` but defaults to the QCNNModel
    defined in the ML reference.  It evaluates a list of observables on each
    parameter set and optionally injects shot‑based Gaussian noise.
    """

    def __init__(self, model: nn.Module | None = None) -> None:
        self.model = model if model is not None else QCNN()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Evaluate the model for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
            Callables that map the model output tensor to a scalar value.
            If empty, the mean of the output is used.
        parameter_sets : Sequence[Sequence[float]]
            A sequence of parameter vectors to feed into the model.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each
            observable value to emulate quantum shot noise.
        seed : int, optional
            Random seed for reproducible noise generation.

        Returns
        -------
        List[List[float]]
            A 2‑D list where each row corresponds to a parameter set and each
            column corresponds to an observable evaluation.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)

                if shots is not None:
                    rng = np.random.default_rng(seed)
                    row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]

                results.append(row)

        return results


__all__ = ["FastHybridEstimator"]
