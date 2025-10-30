"""Advanced base estimator for classical neural networks with batched inference and noise control."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable, Iterable, List, Sequence, Union, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(
    values: Union[Sequence[Sequence[float]], torch.Tensor],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert a list of parameter sequences or a torch tensor into a 2-D tensor."""
    if isinstance(values, torch.Tensor):
        tensor = values
    else:
        tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


class AdvancedBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables with optional noise."""

    def __init__(self, model: nn.Module, device: torch.device | None = None) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]] | torch.Tensor,
        *,
        shots: int | None = None,
        seed: int | None = None,
        return_tensor: bool = False,
    ) -> List[List[float]] | torch.Tensor:
        """Return mean values for each observable and parameter set.

        Parameters
        ----------
        observables:
            Callables that map the model output tensor to a scalar or tensor.
        parameter_sets:
            Iterable of parameter sequences or a tensor of shape (batch, *params).
        shots:
            If provided, Gaussian shot noise with variance 1/shots is added.
        seed:
            Random seed for reproducible noise.
        return_tensor:
            If True, return a torch.Tensor of shape (batch, num_obs).
        """
        self.model.eval()
        with torch.no_grad():
            inputs = _ensure_batch(parameter_sets, device=self.device)
            outputs = self.model(inputs)  # shape (batch,...)

            batch_size = outputs.shape[0]
            observables = list(observables)
            num_obs = len(observables)
            results = np.empty((batch_size, num_obs), dtype=np.float32)

            for idx, obs in enumerate(observables):
                val = obs(outputs)
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                else:
                    val = np.asarray(val)
                if val.ndim == 0:
                    val = np.full((batch_size,), float(val))
                results[:, idx] = val

            if shots is not None:
                rng = np.random.default_rng(seed)
                noise = rng.normal(scale=np.sqrt(1.0 / shots), size=results.shape)
                results = results + noise

            if return_tensor:
                return torch.from_numpy(results).to(self.device)
            else:
                return results.tolist()

    def evaluate_batch_loader(
        self,
        observables: Iterable[ScalarObservable],
        data_loader: DataLoader,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Convenience wrapper to evaluate on a DataLoader."""
        all_results: List[List[float]] = []
        for batch in data_loader:
            batch_results = self.evaluate(
                observables,
                batch,
                shots=shots,
                seed=seed,
                return_tensor=False,
            )
            all_results.extend(batch_results)
        return all_results


class AdvancedEstimator(AdvancedBaseEstimator):
    """Adds Gaussian shot noise on top of deterministic evaluation."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]] | torch.Tensor,
        *,
        shots: int | None = None,
        seed: int | None = None,
        return_tensor: bool = False,
    ) -> List[List[float]] | torch.Tensor:
        raw = super().evaluate(
            observables,
            parameter_sets,
            shots=None,
            seed=None,
            return_tensor=True,
        )
        if shots is None:
            return raw.cpu().tolist() if not return_tensor else raw
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, np.sqrt(1.0 / shots), size=raw.shape)
        noisy = raw + torch.from_numpy(noise).to(self.device)
        return noisy.cpu().tolist() if not return_tensor else noisy


__all__ = ["AdvancedBaseEstimator", "AdvancedEstimator"]
