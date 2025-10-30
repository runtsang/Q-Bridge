"""Hybrid estimator with optional Pennylane QNode backend and classical surrogate support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import pennylane as qml
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor of shape (batch, 1) for any 1‑D input sequence."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor.unsqueeze_(0)
    return tensor


class FastHybridEstimator:
    """
    Evaluates expectation values using either a Pennylane QNode or a classical
    surrogate neural network.  The class is intentionally lightweight to
    facilitate rapid prototyping and benchmarking.

    Parameters
    ----------
    qnode : Optional[qml.QNode]
        Pennylane QNode that maps a parameter vector to a state vector or
        measurement tensor.  If ``None`` the surrogate model is used.
    surrogate : Optional[nn.Module]
        Classical surrogate that approximates the quantum circuit.  Required
        if ``qnode`` is ``None``.
    """

    def __init__(
        self,
        qnode: Optional[qml.QNode] = None,
        surrogate: Optional[nn.Module] = None,
    ) -> None:
        if qnode is None and surrogate is None:
            raise ValueError("Either a qnode or a surrogate model must be provided.")
        self.qnode = qnode
        self.surrogate = surrogate

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Functions that map a model output tensor to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            List of parameter vectors.
        shots : Optional[int]
            If provided, add Gaussian shot noise with variance ``1/shots``.
        seed : Optional[int]
            Random seed for noise generation.

        Returns
        -------
        List[List[float]]
            Nested list of scalar results.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        for params in parameter_sets:
            if self.qnode is not None:
                # Pennylane QNode returns a torch tensor of shape (output_dim,)
                outputs = self.qnode(params)
            else:
                # Use surrogate model
                batch = _ensure_batch(params)
                outputs = self.surrogate(batch).squeeze(0)

            row: List[float] = []
            for obs in observables:
                value = obs(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = float(value.mean().cpu())
                else:
                    scalar = float(value)
                row.append(scalar)
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastHybridEstimator"]
