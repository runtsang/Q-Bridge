"""Enhanced FastBaseEstimator for PyTorch models with batched inference, GPU support, differentiable observables,
shot‑noise simulation, and analytic gradients."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats into a 2‑D float32 tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for multiple parameter sets.

    Parameters
    ----------
    model : nn.Module
        Any differentiable PyTorch model.
    device : str | torch.device, default="cpu"
        Execution device (CPU or CUDA).

    Notes
    -----
    * ``evaluate`` is backward compatible with the original seed.
    * ``evaluate_batch`` supports optional batching for speed.
    * ``compute_gradients`` returns analytic gradients of the first observable.
    * ``evaluate_with_shots`` adds Poissonian shot noise to emulate quantum measurements.
    """

    def __init__(self, model: nn.Module, device: Union[str, torch.device] = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    # --------------------------------------------------------------------- #
    # Core evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return observable values for each parameter set.

        Parameters
        ----------
        observables : Iterable of callables
            Each callable receives the model output and returns a scalar or a tensor.
        parameter_sets : Sequence of parameter vectors

        Returns
        -------
        List[List[float]]
            A list of rows; each row contains the observable values for a parameter set.
        """
        self.model.eval()
        results: List[List[float]] = []
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

    # --------------------------------------------------------------------- #
    # Batched evaluation
    # --------------------------------------------------------------------- #
    def evaluate_batch(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        batch_size: int = 64,
    ) -> List[List[float]]:
        """Evaluate in mini‑batches for large parameter sets.

        Parameters
        ----------
        batch_size : int
            Number of parameter vectors to process per batch.

        Returns
        -------
        List[List[float]]
            Same shape as ``evaluate``.
        """
        self.model.eval()
        results: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, len(parameter_sets), batch_size):
                batch = _ensure_batch(parameter_sets[start : start + batch_size]).to(self.device)
                outputs = self.model(batch)
                for i in range(batch.shape[0]):
                    row: List[float] = []
                    for observable in observables:
                        value = observable(outputs[i : i + 1])
                        if isinstance(value, torch.Tensor):
                            scalar = float(value.mean().cpu())
                        else:
                            scalar = float(value)
                        row.append(scalar)
                    results.append(row)
        return results

    # --------------------------------------------------------------------- #
    # Gradient computation
    # --------------------------------------------------------------------- #
    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        observable_index: int = 0,
    ) -> List[List[float]]:
        """Return gradients of the specified observable w.r.t. model parameters.

        Parameters
        ----------
        observable_index : int
            Index of the observable in ``observables`` whose gradient is requested.

        Returns
        -------
        List[List[float]]
            Each row corresponds to the flattened gradient vector for a parameter set.
        """
        self.model.train()
        grads_list: List[List[float]] = []
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            observable = list(observables)[observable_index]
            value = observable(outputs)
            if isinstance(value, torch.Tensor):
                value = value.mean()
            loss = value
            loss.backward()
            grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.detach().cpu().numpy().flatten())
            grads_list.append(np.concatenate(grads).tolist())
            self.model.zero_grad(set_to_none=True)
        return grads_list

    # --------------------------------------------------------------------- #
    # Shot‑noise simulation
    # --------------------------------------------------------------------- #
    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Add Poissonian shot noise to the deterministic evaluation.

        Parameters
        ----------
        shots : int
            Number of measurement shots per observable.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        List[List[float]]
            Noisy observable estimates.
        """
        rng = np.random.default_rng(seed)
        raw = self.evaluate(observables, parameter_sets)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.poisson(mean) / shots) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
