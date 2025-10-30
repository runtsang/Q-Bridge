"""Classical RBF kernel estimator with optional shot noise.

The estimator exposes a `kernel_matrix` method that returns the Gram
matrix between two collections of feature vectors.  Internally it
leverages the lightweight FastEstimator pattern from the FastBaseEstimator
pair, allowing the user to optionally inject Gaussian shot noise to
simulate finite‑shots quantum measurements.

The design keeps the classical and quantum implementations
compatible at the API level so that they can be swapped or compared
directly in the same experiment scripts.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class RBFKernel(nn.Module):
    """Pure‑Python RBF kernel implemented as a torch module."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-γ‖x−y‖²) for 1‑D tensors `x` and `y`."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class FastBaseEstimator:
    """Base class that evaluates a callable model for batches of inputs."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
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
    """Adds optional Gaussian shot noise to a deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class QuantumKernelEstimator:
    """Classical RBF kernel wrapped in a FastEstimator interface.

    Parameters
    ----------
    gamma : float, optional
        RBF kernel width.
    shots : int | None, optional
        If provided, Gaussian noise with std 1/√shots is added to each
        kernel evaluation to emulate finite‑shot measurements.
    seed : int | None, optional
        Random seed for reproducibility of the noise.
    """
    def __init__(self, gamma: float = 1.0, shots: Optional[int] = None, seed: Optional[int] = None) -> None:
        self.gamma = gamma
        self.shots = shots
        self.seed = seed
        self.kernel = RBFKernel(gamma)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Return the Gram matrix between `a` and `b`.

        The method constructs a list of parameter pairs `(x, y)` and
        evaluates the kernel via the FastEstimator.  If ``shots`` is
        provided, Gaussian noise is added to each entry.
        """
        # Use the supplied shots/seed if provided, otherwise fall back to the
        # instance defaults.
        shots = shots if shots is not None else self.shots
        seed = seed if seed is not None else self.seed

        # Build the parameter set: each element is a 2‑D tensor (x, y).
        param_sets: List[Sequence[float]] = []
        for x in a:
            for y in b:
                # Concatenate x and y to feed into the kernel as a single
                # input vector; this mirrors the FastEstimator pattern.
                param_sets.append(torch.cat([x, y]).tolist())

        estimator = FastEstimator(self.kernel, shots=shots, seed=seed)
        # Observables are identity: we want the raw kernel value.
        results = estimator.evaluate([lambda out: out], param_sets)
        matrix = np.array([row[0] for row in results]).reshape(len(a), len(b))
        return matrix


__all__ = ["RBFKernel", "FastBaseEstimator", "FastEstimator", "QuantumKernelEstimator"]
