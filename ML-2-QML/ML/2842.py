from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ClassicalSelfAttention:
    """Pure‑Python self‑attention used as a feature extractor."""

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class HybridBaseEstimator:
    """Hybrid estimator for classical neural networks with optional self‑attention and shot noise."""

    def __init__(
        self,
        model: nn.Module,
        *,
        use_self_attention: bool = False,
        sa_params: dict | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.use_self_attention = use_self_attention
        self.shots = shots
        self.seed = seed
        if use_self_attention:
            if sa_params is None:
                raise ValueError("sa_params must be provided when use_self_attention is True")
            self.sa = ClassicalSelfAttention(embed_dim=sa_params["embed_dim"])
            self.sa_rotation = sa_params["rotation_params"]
            self.sa_entangle = sa_params["entangle_params"]
        else:
            self.sa = None

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
                if self.use_self_attention:
                    inputs_np = inputs.numpy()
                    inputs = torch.as_tensor(
                        self.sa.run(self.sa_rotation, self.sa_entangle, inputs_np),
                        dtype=torch.float32,
                    )
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
        if self.shots is None:
            return results
        rng = np.random.default_rng(self.seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridBaseEstimator"]
