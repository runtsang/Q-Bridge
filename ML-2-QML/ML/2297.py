"""Hybrid fast estimator for classical neural networks with optional shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of scalars to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFastEstimator:
    """
    Lightweight estimator for PyTorch models.

    Parameters
    ----------
    model : nn.Module
        Any neural network that accepts a 2‑D tensor of shape (batch, features)
        and returns a 2‑D tensor of outputs.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute the mean of each observable over the model outputs.

        Parameters
        ----------
        observables
            Callables that map the model output tensor to a scalar or tensor.
        parameter_sets
            Iterable of parameter vectors to feed into the model.
        shots
            If supplied, Gaussian shot noise with variance 1/shots is added.
        seed
            Random seed for reproducibility.

        Returns
        -------
        List[List[float]]
            A list of rows, one per parameter set, each containing the
            evaluated observables.
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
                    scalar = (
                        float(value.mean().cpu())
                        if isinstance(value, torch.Tensor)
                        else float(value)
                    )
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                rng.normal(mean, max(1e-6, 1 / shots)) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------
    # Convenience wrappers for the classical FCL example
    # ------------------------------------------------------------------
    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run the fully‑connected layer when the underlying model provides a
        ``run`` attribute (as the FCL helper does).

        Parameters
        ----------
        thetas
            Iterable of parameters to feed into the model.

        Returns
        -------
        np.ndarray
            The scalar expectation returned by the model.
        """
        if not hasattr(self.model, "run"):
            raise AttributeError("Model does not expose a `run` method.")
        return self.model.run(thetas)

__all__ = ["HybridFastEstimator"]
