"""Hybrid classical model inspired by Quantum‑NAT, enriched with quantum‑inspired layers and fast evaluation utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class QuantumNATHybrid(nn.Module):
    """Classical CNN + FC with a quantum‑inspired feature layer."""

    class QuantumInspiredLayer(nn.Module):
        """A lightweight layer mimicking a quantum circuit using random linear maps and trainable gates."""

        def __init__(self, input_dim: int, hidden_dim: int = 32):
            super().__init__()
            # Fixed random orthogonal projection
            self.register_buffer(
                "random_proj",
                torch.randn(input_dim, hidden_dim) / np.sqrt(input_dim),
            )
            # Trainable linear transformation simulating parameterized gates
            self.lin = nn.Linear(hidden_dim, hidden_dim)
            self.b = nn.Parameter(torch.zeros(hidden_dim))
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (bsz, input_dim)
            proj = x @ self.random_proj  # fixed random projection
            gated = torch.sin(self.lin(proj) + self.b)  # sine to mimic angle‑encoding
            return self.norm(gated)

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Feature dimension after conv: 16 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.quantum_layer = self.QuantumInspiredLayer(64, 32)
        self.out = nn.Linear(32, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        hidden = self.fc(flat)
        quantum = self.quantum_layer(hidden)
        out = self.out(quantum)
        return self.norm(out)


class FastBaseEstimator:
    """Fast deterministic estimator for the classical model."""

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
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
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
    """Estimator with optional shot‑noise emulation."""

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
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["QuantumNATHybrid", "FastBaseEstimator", "FastEstimator"]
