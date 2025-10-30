"""Hybrid estimator combining PyTorch models with optional classical self‑attention."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Ensure a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ClassicalSelfAttention:
    """Simple single‑head self‑attention compatible with the FastBaseEstimator interface."""

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def __call__(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class FastHybridEstimator:
    """Evaluate a PyTorch network and optionally a classical self‑attention block."""

    def __init__(
        self,
        model: nn.Module,
        attention: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
        *,
        noise_shots: int | None = None,
        noise_seed: int | None = None,
    ) -> None:
        self.model = model
        self.attention = attention
        self.noise_shots = noise_shots
        self.noise_seed = noise_seed

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
                if self.attention is not None:
                    # apply classical self‑attention before feeding to the model
                    attn_out = self.attention(
                        rotation_params=np.random.randn(inputs.shape[-1], inputs.shape[-1]),
                        entangle_params=np.random.randn(inputs.shape[-1]),
                        inputs=inputs.numpy(),
                    )
                    inputs = _ensure_batch(attn_out)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                results.append(row)

        if self.noise_shots is None:
            return results

        rng = np.random.default_rng(self.noise_seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.noise_shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastHybridEstimator", "ClassicalSelfAttention"]
