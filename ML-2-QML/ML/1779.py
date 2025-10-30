"""Enhanced FastBaseEstimator with GPU support, shot‑noise simulation and gradient capability.

This module introduces:
  * Automatic device selection (CPU/GPU).
  * Optional Gaussian shot noise to mimic finite‑shot measurements.
  * A separate `gradients` method that returns the analytic norm of the gradient of
    each observable with respect to the model parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class FastBaseEstimatorPro:
    """Enhanced estimator for PyTorch neural networks."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            The PyTorch model to evaluate.
        device : Optional[torch.device]
            Device to run the model on. If None, CUDA if available, else CPU.
        """
        self.model = model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = self.model.device

    def _ensure_batch(self, values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute scalar observables for each parameter set, optionally adding Gaussian shot noise.

        Parameters
        ----------
        observables : Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
            Functions mapping model outputs to a scalar or a tensor.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors for the model.
        shots : Optional[int]
            Number of Monte‑Carlo shots to add Gaussian noise.
        seed : Optional[int]
            Random seed for reproducibility.

        Returns
        -------
        List[List[float]]
            For each parameter set, a list containing the observable values.
        """
        self.model.eval()
        rng = np.random.default_rng(seed)
        results: List[List[float]] = []

        for params in parameter_sets:
            with torch.no_grad():
                inputs = self._ensure_batch(params)
                outputs = self.model(inputs)
            row: List[float] = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val_scalar = float(val.mean().cpu())
                else:
                    val_scalar = float(val)
                if shots is not None:
                    noise = rng.normal(0, max(1e-6, 1 / np.sqrt(shots)))
                    val_scalar += noise
                row.append(val_scalar)
            results.append(row)
        return results

    def gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute the norm of the gradient of each observable with respect to the model
        parameters for every parameter set.

        Parameters
        ----------
        observables : Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
            Functions mapping model outputs to a scalar or a tensor.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors for the model.

        Returns
        -------
        List[List[float]]
            For each parameter set, a list containing the gradient norms of each observable.
        """
        self.model.train()
        grad_results: List[List[float]] = []

        for params in parameter_sets:
            inputs = self._ensure_batch(params)
            self.model.zero_grad()
            outputs = self.model(inputs)
            row: List[float] = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val_sum = val.sum()
                else:
                    val_sum = torch.tensor(val, device=self.device, dtype=torch.float32)
                val_sum.backward(retain_graph=True)
                # Compute norm over all parameter gradients
                norm_sq = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        norm_sq += p.grad.norm().item() ** 2
                row.append(float(norm_sq ** 0.5))
                self.model.zero_grad()
            grad_results.append(row)
        return grad_results


__all__ = ["FastBaseEstimatorPro"]
