"""Enhanced estimator utilities implemented with PyTorch modules.

The class now:
* Evaluates arbitrary callable observables on batched inputs.
* Supports GPU execution when available.
* Injects Gaussian or Poisson shot noise.
* Computes analytical gradients of observables w.r.t. model parameters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.autograd import grad

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""

    def __init__(self, model: nn.Module, device: Optional[str] = None) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        return_tensors: bool = False,
    ) -> List[List[float]] | List[torch.Tensor]:
        """Return a list of lists of observable values for each parameter set."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
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
        if return_tensors:
            return [torch.tensor(row, device=self.device) for row in results]
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        noise_type: str = "gaussian",
        seed: int | None = None,
    ) -> List[List[float]]:
        """Add shot‑noise to deterministic evaluations."""
        raw = self.evaluate(observables, parameter_sets, return_tensors=False)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            if noise_type == "gaussian":
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            elif noise_type == "poisson":
                noisy_row = [float(rng.poisson(mean * shots) / shots) for mean in row]
            else:
                raise ValueError(f"Unsupported noise_type: {noise_type}")
            noisy.append(noisy_row)
        return noisy

    def gradient(
        self,
        observable: ScalarObservable,
        parameter_set: Sequence[float],
    ) -> torch.Tensor:
        """Return the gradient of a scalar observable w.r.t. model parameters."""
        self.model.train()
        params = torch.tensor(parameter_set, dtype=torch.float32, requires_grad=True, device=self.device)
        outputs = self.model(params)
        value = observable(outputs)
        if isinstance(value, torch.Tensor):
            value = value.mean()
        grads = grad(value, self.model.parameters(), retain_graph=True, create_graph=True)
        return torch.cat([g.reshape(-1) for g in grads]).detach()


__all__ = ["FastBaseEstimator"]
