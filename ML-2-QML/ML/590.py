"""FastEstimator: extend lightweight estimator with gradient support and noise simulation.

The class accepts a PyTorch nn.Module and evaluates scalar observables over
batches of parameter vectors.  It now also offers analytic gradients via
torch.autograd, and can inject Gaussian shot noise to emulate measurement
statistics.  The API mirrors the original seed but adds the ``evaluate_gradients``
method and optional device selection.

Typical usage:

>>> import torch
>>> model = torch.nn.Linear(1, 1)
>>> estimator = FastEstimator(model)
>>> obs = [lambda out: out.mean()]
>>> params = [[0.5], [1.0]]
>>> values = estimator.evaluate(obs, params)
>>> grads = estimator.evaluate_gradients(obs, params)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D list of floats to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Evaluate a PyTorch model for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to be evaluated.  The module must accept a
        batch‑tensor of shape ``(batch, input_dim)`` and return a
        batch‑tensor of outputs.
    device : torch.device, optional
        Device on which to perform computations.  Defaults to ``torch.device('cpu')``.
    """

    def __init__(self, model: nn.Module, device: torch.device | None = None) -> None:
        self.model = model.to(device or torch.device("cpu"))
        self.device = device or torch.device("cpu")

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return expectation values for each observable and parameter set.

        Parameters
        ----------
        observables
            Iterable of callable observables that map a model output tensor to a
            scalar (or a tensor of scalars).  If empty, the mean of the output
            is returned.
        parameter_sets
            Sequence of parameter vectors; each vector is a list/tuple of floats.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
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
                results.append(row)
        return results

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------
    def evaluate_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[np.ndarray]]:
        """Return analytic gradients of each observable w.r.t the input parameters.

        The gradients are returned as NumPy arrays of shape ``(len(params),)``.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        grad_results: List[List[np.ndarray]] = []

        self.model.eval()
        for params in parameter_sets:
            # Create a tensor that requires gradients
            inputs = _ensure_batch(params).to(self.device).requires_grad_(True)
            outputs = self.model(inputs)

            row_grads: List[np.ndarray] = []
            for observable in observables:
                # Compute scalar observable
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, dtype=torch.float32, device=self.device)
                # Gradient of scalar w.r.t inputs
                grad = torch.autograd.grad(scalar, inputs, retain_graph=True)[0]
                row_grads.append(grad.cpu().numpy().flatten())
            grad_results.append(row_grads)
        return grad_results

    # ------------------------------------------------------------------
    # Shot‑noise augmentation
    # ------------------------------------------------------------------
    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Wrap :meth:`evaluate` with Gaussian shot noise.

        Parameters
        ----------
        shots
            Number of simulated measurement shots.  If ``None`` (default) the
            deterministic result is returned.
        seed
            Random seed for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastEstimator"]
