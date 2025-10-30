"""Enhanced lightweight estimator utilities built on PyTorch.

Features
--------
- Automatic device selection (CPU or GPU).
- Batch evaluation of multiple parameter sets.
- Vectorized observables accepting batched outputs.
- Optional gradient computation via autograd.
- Configurable noise injection (Gaussian or Poisson).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
GradientFunction = Callable[[torch.Tensor], torch.Tensor]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    device : str | torch.device, optional
        Compute device; defaults to CUDA if available.
    """

    def __init__(self, model: nn.Module, device: str | torch.device | None = None) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(inputs.to(self.device))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that map model outputs to scalars.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.

        Returns
        -------
        List[List[float]]
            Observables per parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        # Batch evaluation
        batch = torch.stack([torch.as_tensor(p, dtype=torch.float32) for p in parameter_sets])
        outputs = self._forward(batch)

        for obs in observables:
            values = obs(outputs)
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            else:
                values = np.asarray(values)
            if values.ndim > 1:
                values = values.reshape(-1)
            if not results:
                results = [[float(v)] for v in values]
            else:
                for row, v in zip(results, values):
                    row.append(float(v))
        return results

    def evaluate_with_grad(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        grad_fn: GradientFunction | None = None,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Evaluate observables and optionally compute gradients.

        Parameters
        ----------
        grad_fn : GradientFunction, optional
            Function that maps model outputs to gradient tensors. If None,
            autograd will be used.

        Returns
        -------
        Tuple[List[List[float]], List[List[float]]]
            Observables and gradients per parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]

        batch = torch.stack([torch.as_tensor(p, dtype=torch.float32) for p in parameter_sets])
        batch.requires_grad_(True)
        outputs = self.model(batch.to(self.device))

        obs_vals = []
        grads = []

        for obs in observables:
            val = obs(outputs)
            if isinstance(val, torch.Tensor):
                val = val.mean()
            else:
                val = torch.tensor(val, dtype=torch.float32, device=self.device)
            obs_vals.append(val.item())

            if grad_fn is not None:
                grad = grad_fn(outputs)
            else:
                grad = torch.autograd.grad(val, batch, retain_graph=True, allow_unused=True)[0]
            grads.append(grad.detach().cpu().numpy().tolist())

        return obs_vals, grads


class FastEstimator(FastBaseEstimator):
    """Adds configurable noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        noise_type: str = "gaussian",
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            if noise_type == "gaussian":
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            elif noise_type == "poisson":
                noisy_row = [float(rng.poisson(mean, 1)) for mean in row]
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
