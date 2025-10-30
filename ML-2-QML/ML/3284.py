"""Combined classical estimator with optional shot noise and sampler network.

This module defines FastBaseEstimatorGen224 that can evaluate a PyTorch
neural network for multiple parameter sets and observables.
It also exposes a lightweight SamplerQNN class that can be used as a
classification head.

The design combines ideas from the original FastBaseEstimator and
SamplerQNN seeds, adding Gaussian shot noise and a more flexible
observable interface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class SamplerQNN(nn.Module):
    """Simple feed‑forward sampler network used as a classification head.

    Mirrors the structure of the quantum SamplerQNN but implemented
    with PyTorch. The network outputs a probability vector via softmax.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class FastBaseEstimatorGen224:
    """Evaluate a PyTorch model for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    noise_std : float | None, optional
        Standard deviation of Gaussian noise added to each output.
        If ``None`` no noise is added.  This mimics shot noise.
    seed : int | None, optional
        Random seed used for reproducibility of the noise.
    """
    def __init__(
        self,
        model: nn.Module,
        *,
        noise_std: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : Iterable[ScalarObservable]
            Callables that transform a model output into a scalar.
            If empty, the mean of the output is used.
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence contains the parameters to feed to the model.

        Returns
        -------
        List[List[float]]
            A 2‑D list where each inner list contains the values of the
            observables for a single parameter set.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                if self.noise_std is not None:
                    noise = torch.randn_like(outputs) * self.noise_std
                    outputs = outputs + noise
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


__all__ = ["FastBaseEstimatorGen224", "SamplerQNN"]
