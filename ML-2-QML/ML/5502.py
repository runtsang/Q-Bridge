"""Hybrid kernel and estimator framework for classical data."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


class ClassicalKernel(nn.Module):
    """Radial Basis Function kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(ax, by).item() for by in b] for ax in a])


class FastBaseEstimator:
    """Deterministic batched evaluator for PyTorch models."""

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
    """Estimator with optional Gaussian shot noise."""

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


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch convolution followed by flattening."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out.view(x.size(0), -1)


class SelfAttentionModule(nn.Module):
    """Self‑attention layer mirroring the quantum interface."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(inputs)
        k = self.key_proj(inputs)
        v = self.value_proj(inputs)
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class QuantumKernelMethod:
    """Hybrid kernel framework that supports classical RBF and quantum feature maps."""

    def __init__(self, mode: str = "classical", gamma: float = 1.0) -> None:
        self.mode = mode
        if mode == "classical":
            self.kernel = ClassicalKernel(gamma)
        else:
            raise NotImplementedError("Quantum mode requires the quantum module.")

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return self.kernel.matrix(a, b)

    def evaluate(
        self,
        model: nn.Module,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        estimator = FastEstimator(model)
        return estimator.evaluate(observables, parameter_sets)


__all__ = [
    "QuantumKernelMethod",
    "ClassicalKernel",
    "FastEstimator",
    "QuanvolutionFilter",
    "SelfAttentionModule",
]
