import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Iterable
from collections.abc import Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D parameter list into a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class SelfAttention(nn.Module):
    """
    Classical self‑attention module implemented with PyTorch.

    Parameters are split into *rotation* and *entangle* groups.
    The rotation matrix maps the input into query/key/value space,
    while the entangle matrix controls the attention weights.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> torch.Tensor:
        if rotation_params is not None:
            self.rotation.data.copy_(
                torch.as_tensor(rotation_params.reshape(self.embed_dim, self.embed_dim))
            )
        if entangle_params is not None:
            self.entangle.data.copy_(
                torch.as_tensor(entangle_params.reshape(self.embed_dim, self.embed_dim))
            )
        q = inputs @ self.rotation
        k = inputs @ self.entangle
        v = inputs
        scores = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class FastBaseEstimator:
    """
    Evaluate a PyTorch model for a batch of parameter sets.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                rot, ent = params
                # Create a dummy input that matches the embed_dim
                inputs = _ensure_batch(np.arange(self.model.embed_dim))
                outputs = self.model(inputs, rotation_params=rot, entangle_params=ent)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().item())
                    row.append(val)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """
    Adds Gaussian shot noise to the deterministic estimates.
    """

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
        noisy = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["SelfAttention", "FastBaseEstimator", "FastEstimator"]
