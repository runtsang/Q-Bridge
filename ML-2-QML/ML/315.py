"""Enhanced FastBaseEstimator for PyTorch models with GPU support, shot‑noise simulation and gradient computation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of parameters into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimatorGen350:
    """Evaluate a PyTorch model for many parameter sets and observables.

    Parameters
    ----------
    model
        Any :class:`torch.nn.Module` that maps a batch of parameter vectors to
        output tensors.
    device
        Optional device specification.  If ``None`` the model's device is used.

    Features
    --------
    * Vectorised evaluation on GPU if available.
    * Supports custom scalar observables that may return tensors.
    * Optional shot‑noise simulation with Gaussian or Poisson statistics.
    * Automatic differentiation to compute gradients w.r.t. model parameters.
    """

    def __init__(self, model: nn.Module, device: Optional[Union[str, torch.device]] = None) -> None:
        self.model = model.to(device or torch.device("cpu"))
        self.device = self.model.device

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        noise: str = "none",
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Evaluate the model for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of callables mapping the model output to a scalar.
        parameter_sets
            Iterable of parameter vectors.
        shots
            If provided, simulates finite‑shot readout.
        noise
            ``"none"``, ``"gaussian"``, or ``"poisson"`` – determines how shot noise is added.
        seed
            Random seed for reproducibility of shot noise.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        rng = np.random.default_rng(seed)

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                outputs = self._forward(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)

                if shots is not None:
                    if noise == "gaussian":
                        std = max(1e-6, 1.0 / shots)
                        row = [rng.normal(mean, std) for mean in row]
                    elif noise == "poisson":
                        row = [rng.poisson(mean * shots) / shots for mean in row]
                results.append(row)
        return results

    def gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[np.ndarray]]]:
        """
        Compute gradients of each observable w.r.t. model parameters for every
        parameter set.  Returns a nested list, where the outer list corresponds
        to the parameter sets, the middle list to observables, and the innermost
        list contains NumPy arrays for each model parameter.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        grads: List[List[List[np.ndarray]]] = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)
            outputs = self._forward(inputs)
            row_grads: List[List[np.ndarray]] = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                grad_tensors = torch.autograd.grad(val, self.model.parameters(), retain_graph=True)
                grad_arrays = [g.detach().cpu().numpy() if g is not None else None for g in grad_tensors]
                row_grads.append(grad_arrays)
            grads.append(row_grads)
        return grads


__all__ = ["FastBaseEstimatorGen350"]
