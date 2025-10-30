"""Hybrid estimator for classical neural networks with optional noise and fully connected layer support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """Evaluate a PyTorch model for batches of inputs and observables."""

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
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    @staticmethod
    def FCL(n_features: int = 1) -> nn.Module:
        """Return a simple fully‑connected layer mimicking the quantum FCL example."""
        class FullyConnectedLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(n_features, 1)

            def run(self, thetas: Iterable[float]) -> np.ndarray:
                values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
                expectation = torch.tanh(self.linear(values)).mean(dim=0)
                return expectation.detach().cpu().numpy()

        return FullyConnectedLayer()

    @staticmethod
    def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    @staticmethod
    def RegressionDataset(samples: int, num_features: int) -> nn.Module:
        from torch.utils.data import Dataset

        class _RegressionDataset(Dataset):
            def __init__(self) -> None:
                self.features, self.labels = HybridEstimator.generate_superposition_data(num_features, samples)

            def __len__(self) -> int:  # type: ignore[override]
                return len(self.features)

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
                return {
                    "states": torch.tensor(self.features[idx], dtype=torch.float32),
                    "target": torch.tensor(self.labels[idx], dtype=torch.float32),
                }

        return _RegressionDataset()

    @staticmethod
    def QModel(num_features: int) -> nn.Module:
        class _QModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(num_features, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )

            def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                return self.net(state_batch.to(torch.float32)).squeeze(-1)

        return _QModel()

__all__ = ["HybridEstimator"]
