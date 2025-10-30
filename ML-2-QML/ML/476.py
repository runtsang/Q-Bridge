"""Enhanced estimator for PyTorch neural networks with batch processing, GPU support, shot noise, and gradient evaluation."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Base class that evaluates a PyTorch model for given parameter sets."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate observables for each parameter set. Returns list of lists."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
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


class FastEstimator(FastBaseEstimator):
    """Extended estimator that supports GPU, batched evaluation, optional shot noise, and gradients."""
    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(model)
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    def _evaluate_batch(
        self,
        observables: List[ScalarObservable],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Internal helper to compute observables on a batch of inputs."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch.to(self.device))
            batch_results = []
            for obs in observables:
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.mean(dim=-1)
                batch_results.append(val.cpu())
            return torch.stack(batch_results, dim=1)  # (batch_size, n_obs)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int | None = None,
        return_tensor: bool = False,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]] | torch.Tensor:
        """Evaluate observables with optional shot noise and batched processing."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        n_obs = len(observables)
        n_params = len(parameter_sets)

        all_params = torch.tensor(parameter_sets, dtype=torch.float32, device=self.device)

        if batch_size is None or batch_size >= n_params:
            batch_results = self._evaluate_batch(observables, all_params)
        else:
            batch_results = []
            for start in range(0, n_params, batch_size):
                end = start + batch_size
                chunk = all_params[start:end]
                batch_results.append(self._evaluate_batch(observables, chunk))
            batch_results = torch.cat(batch_results, dim=0)

        batch_results = batch_results.cpu()
        if shots is not None:
            rng = np.random.default_rng(seed)
            noise = rng.normal(loc=0.0, scale=np.sqrt(1.0 / shots), size=batch_results.shape)
            batch_results += noise

        if return_tensor:
            return batch_results

        return batch_results.tolist()

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return observable values and gradients w.r.t. input parameters."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        n_obs = len(observables)
        all_params = torch.tensor(parameter_sets, dtype=torch.float32, device=self.device, requires_grad=True)

        outputs = self.model(all_params)
        obs_tensors = []
        for obs in observables:
            val = obs(outputs)
            if isinstance(val, torch.Tensor):
                val = val.mean(dim=-1)
            obs_tensors.append(val)

        values = torch.stack(obs_tensors, dim=1)
        grads = torch.autograd.grad(
            outputs=values,
            inputs=all_params,
            grad_outputs=torch.ones_like(values),
            create_graph=False,
            retain_graph=False,
        )[0]

        return values.detach(), grads.detach()

    def predict(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Return raw model outputs for given parameter sets."""
        all_params = torch.tensor(parameter_sets, dtype=torch.float32, device=self.device)
        if batch_size is None or batch_size >= len(parameter_sets):
            return self.model(all_params).cpu()
        else:
            outputs = []
            for start in range(0, len(parameter_sets), batch_size):
                end = start + batch_size
                chunk = all_params[start:end]
                outputs.append(self.model(chunk).cpu())
            return torch.cat(outputs, dim=0)

__all__ = ["FastBaseEstimator", "FastEstimator"]
