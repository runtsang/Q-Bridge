"""Hybrid estimator that unifies classical neural‑network evaluation with optional quantum‑inspired features.

The module supports:
* deterministic evaluation of torch.nn.Module models,
* optional Gaussian shot noise to emulate measurement uncertainty,
* optional self‑attention layer that can be prepended to the network,
* convenient factory functions for EstimatorQNN and SamplerQNN.

The API mirrors the original FastBaseEstimator but extends it with a
self‑attention wrapper and noise injection.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], Union[torch.Tensor, float]]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of parameters to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class SelfAttention(nn.Module):
    """Classical self‑attention wrapper.

    The layer applies a linear projection to the inputs to obtain query,
    key, and value tensors and then performs a scaled‑dot‑product
    attention.  It is lightweight and can be inserted before an
    arbitrary neural network.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class HybridEstimator:
    """Hybrid estimator that works with torch.nn.Module models.

    Parameters
    ----------
    model : nn.Module
        The underlying neural network.
    noise_shots : int | None, optional
        If provided, Gaussian noise with std = 1 / sqrt(noise_shots) is added
        to each observable mean to emulate measurement shot noise.
    noise_seed : int | None, optional
        Seed for the noise generator.
    self_attention : bool, optional
        Prepend a ``SelfAttention`` layer to ``model`` if ``True``.
    attention_dim : int, optional
        Dimensionality of the self‑attention projections; used only when
        ``self_attention`` is ``True``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        noise_shots: Optional[int] = None,
        noise_seed: Optional[int] = None,
        self_attention: bool = False,
        attention_dim: int = 4,
    ) -> None:
        self.model = model
        self.noise_shots = noise_shots
        self.noise_seed = noise_seed
        self.rng = np.random.default_rng(noise_seed)

        if self_attention:
            # prepend a self‑attention block
            sa = SelfAttention(attention_dim)
            # wrap the original model so that attention is applied first
            original_forward = self.model.forward

            def new_forward(x: torch.Tensor) -> torch.Tensor:
                return original_forward(sa(x))

            self.model.forward = new_forward  # type: ignore[assignment]

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate the model and return a list of observable values.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a
            scalar tensor or scalar.
        parameter_sets : sequence of float sequences
            Each inner sequence corresponds to a set of input parameters.

        Returns
        -------
        List[List[float]]
            Outer list over parameter sets, inner list over observables.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = _ensure_batch(params)
                out = self.model(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)

        if self.noise_shots is not None:
            noisy: List[List[float]] = []
            std = max(1e-6, 1 / np.sqrt(self.noise_shots))
            for row in results:
                noisy_row = [float(self.rng.normal(mean, std)) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results


def EstimatorQNN() -> nn.Module:
    """Return a simple fully‑connected regression network."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(x)

    return EstimatorNN()


def SamplerQNN() -> nn.Module:
    """Return a simple classifier network producing a probability vector."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return torch.softmax(self.net(x), dim=-1)

    return SamplerModule()


__all__ = ["HybridEstimator", "SelfAttention", "EstimatorQNN", "SamplerQNN"]
