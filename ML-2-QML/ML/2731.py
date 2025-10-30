"""Hybrid estimator combining classical neural network evaluation and sampling.

This module defines :class:`FastHybridEstimator` which can evaluate a PyTorch
neural network for arbitrary parameter sets and observables, optionally add
Gaussian shot noise, and provide a unified sampling interface via a simple
``SamplerQNN`` neural network.  The design mirrors the original
``FastBaseEstimator`` but extends it to support a stochastic sampling
mechanism and a clearer separation between deterministic evaluation and
noisy simulation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, Iterable, List, Sequence, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Ensure a 2‑D batch tensor for the model."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class SamplerQNN(nn.Module):
    """A lightweight neural sampler that maps a 2‑dimensional input to a
    probability distribution over two outcomes.  The architecture is
    intentionally simple to keep evaluation fast while still providing
    non‑trivial non‑linearity.
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


class FastHybridEstimator:
    """Evaluate a PyTorch model for a set of parameters and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.  It must accept a 2‑D tensor of
        shape ``(batch, features)`` and return a tensor of shape
        ``(batch, output_dim)``.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        If ``shots`` is provided, Gaussian noise with variance ``1/shots``
        is added to each mean value to mimic finite‑shot statistics.
        """
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def sample(
        self,
        sampler: SamplerQNN,
        parameter_sets: Sequence[Sequence[float]],
        num_samples: int,
        *,
        seed: Optional[int] = None,
    ) -> List[List[int]]:
        """Draw samples from the neural sampler.

        Parameters
        ----------
        sampler : SamplerQNN
            The neural network that outputs a probability distribution.
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence contains the 2‑dimensional input for the
            sampler.
        num_samples : int
            Number of samples to draw per input.
        seed : int, optional
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        samples: List[List[int]] = []
        sampler.eval()
        with torch.no_grad():
            for params in parameter_sets:
                probs = sampler(_ensure_batch(params)).cpu().numpy().flatten()
                sample = rng.choice(len(probs), size=num_samples, p=probs)
                samples.append(sample.tolist())
        return samples


__all__ = ["FastHybridEstimator", "SamplerQNN"]
