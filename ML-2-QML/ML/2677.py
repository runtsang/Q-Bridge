"""HybridSamplerQNN: classical sampler with fast batched evaluation and optional shot noise.

The module implements a lightweight PyTorch sampler network and a fast estimator
that mirrors the quantum counterpart. It supports batched input/weight
parameter sets, deterministic evaluation, and optional Gaussian shot noise
to emulate finite‑shot sampling. The design follows the same interface as
the quantum FastBaseEstimator, enabling a unified API across classical
and quantum experiments.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of parameters and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
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


class SamplerQNN(nn.Module):
    """Simple 2‑input, 2‑output softmax sampler network."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


class HybridSamplerQNN:
    """Wrapper that exposes the classical sampler and a fast estimator."""

    def __init__(self, *, noise_shots: int | None = None, seed: int | None = None) -> None:
        self.model = SamplerQNN()
        self.estimator = FastEstimator(self.model)
        self.noise_shots = noise_shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate the sampler for given parameters and observables.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar.
        parameter_sets : sequence of parameter sequences
            Each sequence contains the 2 input parameters for the network.

        Returns
        -------
        List[List[float]]
            Values for each observable and parameter set.
        """
        return self.estimator.evaluate(
            observables,
            parameter_sets,
            shots=self.noise_shots,
            seed=self.seed,
        )


__all__ = ["HybridSamplerQNN", "SamplerQNN", "FastEstimator", "FastBaseEstimator"]
