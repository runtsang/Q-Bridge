"""Enhanced fast estimator for classical neural networks with GPU support,
vectorized evaluation, and gradient access."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float], device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of parameters with optional GPU acceleration.

    Parameters
    ----------
    model : nn.Module
        Pre‑trained neural network mapping parameter vectors to outputs.
    device : torch.device | str, optional
        Device on which to run the model (default: ``cpu``).
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.model(params)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int | None = None,
    ) -> List[List[float]]:
        """Compute the mean of each observable over the model output for every parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the raw model output tensor and returns a scalar.
        parameter_sets : sequence of sequences
            Each inner sequence is a list of float parameters.
        batch_size : int, optional
            Number of parameter sets to evaluate in one forward pass (``None`` -> all).

        Returns
        -------
        list[list[float]]
            A 2‑D list where rows correspond to parameter sets and columns to observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        # Vectorised batch handling
        if batch_size is None or batch_size >= len(parameter_sets):
            batch = _ensure_batch(parameter_sets, self.device)
            outputs = self._forward(batch)
            rows = self._process_observables(outputs, observables)
            results.extend(rows)
        else:
            for i in range(0, len(parameter_sets), batch_size):
                batch = _ensure_batch(parameter_sets[i : i + batch_size], self.device)
                outputs = self._forward(batch)
                rows = self._process_observables(outputs, observables)
                results.extend(rows)

        return results

    def _process_observables(
        self, outputs: torch.Tensor, observables: List[ScalarObservable]
    ) -> List[List[float]]:
        rows: List[List[float]] = []
        for out in outputs:
            row: List[float] = []
            for obs in observables:
                value = obs(out)
                if isinstance(value, torch.Tensor):
                    scalar = float(value.mean().cpu())
                else:
                    scalar = float(value)
                row.append(scalar)
            rows.append(row)
        return rows

    def evaluate_with_grad(
        self,
        observables: Iterable[ScalarObservable],
        parameter_set: Sequence[float],
        *,
        batch_size: int | None = None,
    ) -> Tuple[List[float], List[torch.Tensor]]:
        """Return both observable values and their gradients w.r.t the parameters.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the raw model output tensor and returns a scalar.
        parameter_set : sequence of float
            Single set of parameters.
        batch_size : int, optional
            Unused for single set but kept for API compatibility.

        Returns
        -------
        tuple[list[float], list[torch.Tensor]]
            The first element contains the observable values, the second element contains the gradients
            (one gradient tensor per observable) with respect to the input parameters.
        """
        self.model.train()
        params_tensor = _ensure_batch(parameter_set, self.device)
        params_tensor.requires_grad_(True)

        outputs = self._forward(params_tensor)
        obs_values: List[torch.Tensor] = []
        grads: List[torch.Tensor] = []

        for obs in observables:
            val = obs(outputs[0])  # scalar
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, device=self.device)
            val.backward(retain_graph=True)
            grads.append(params_tensor.grad.squeeze().clone())
            obs_values.append(val.detach().cpu())
            params_tensor.grad.zero_()

        self.model.eval()
        return [float(v) for v in obs_values], grads

    def add_noisy_observables(
        self,
        shots: int,
        seed: int | None = None,
    ) -> None:
        """Add Gaussian shot‑noise to subsequent evaluations by wrapping the original
        `evaluate` method. This is a lightweight, optional feature that preserves
        the deterministic API.
        """
        original_evaluate = self.evaluate

        def noisy_evaluate(
            observables: Iterable[ScalarObservable], parameter_sets: Sequence[Sequence[float]]
        ) -> List[List[float]]:
            raw = original_evaluate(observables, parameter_sets)
            rng = np.random.default_rng(seed)
            return [
                [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                for row in raw
            ]

        self.evaluate = noisy_evaluate  # type: ignore

__all__ = ["FastBaseEstimator"]
