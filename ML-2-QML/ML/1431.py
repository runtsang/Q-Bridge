"""Enhanced estimator utilities built on PyTorch with gradient support and device flexibility."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

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


class FastBaseEstimatorExtended:
    """Evaluate neural networks for batches of inputs and observables with optional gradient support.

    This class extends the original lightweight estimator by:
    1. Allowing specification of the computation device (CPU or CUDA).
    2. Supporting autograd to compute gradients of observables w.r.t. model parameters.
    3. Providing a flexible noise model that can be applied to each evaluation.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def _evaluate_batch(
        self,
        observables: Iterable[ScalarObservable],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Internal helper that returns a tensor of shape (batch, num_observables)."""
        outputs = self.model(inputs)
        obs_values: List[torch.Tensor] = []
        for observable in observables:
            val = observable(outputs)
            if isinstance(val, torch.Tensor):
                obs_values.append(val.reshape(-1, 1))
            else:
                obs_values.append(
                    torch.tensor(val, dtype=torch.float32, device=self.device).reshape(-1, 1)
                )
        return torch.cat(obs_values, dim=1)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute deterministic or noisy expectation values for a collection of parameter sets.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that map model outputs to scalar values.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors to evaluate.
        shots : int | None, optional
            If provided, inject Gaussian noise with variance 1/shots.
        seed : int | None, optional
            Seed for reproducible noise generation.
        """
        self.model.eval()
        rng = np.random.default_rng(seed)
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
                batch_vals = self._evaluate_batch(observables, inputs)
                row = batch_vals.squeeze().cpu().numpy().tolist()
                if shots is not None:
                    noise = rng.normal(0, max(1e-6, 1 / shots), size=len(row))
                    row = (np.array(row) + noise).tolist()
                results.append(row)
        return results

    def evaluate_and_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[tuple[List[float], List[torch.Tensor]]]:
        """Return both expectation values and gradients for each parameter set.

        The gradients are returned as a list of tensors, one per observable, each
        having shape (parameter_count,).
        """
        self.model.train()
        results: List[tuple[List[float], List[torch.Tensor]]] = []
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            grads_per_obs: List[torch.Tensor] = []
            val_list: List[float] = []
            for observable in observables:
                val = observable(outputs)
                if isinstance(val, torch.Tensor):
                    scalar = val.mean()
                else:
                    scalar = torch.tensor(val, dtype=torch.float32, device=self.device)
                val_list.append(float(scalar.item()))
                scalar.backward(retain_graph=True)
                grads_per_obs.append(inputs.grad.clone().detach().squeeze().cpu())
                inputs.grad.zero_()
            results.append((val_list, grads_per_obs))
        return results


__all__ = ["FastBaseEstimatorExtended"]
