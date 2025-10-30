"""
Classical regression infrastructure with a fully‑connected layer and estimator utilities.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Iterable, Sequence, List, Callable, Any

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.  Each sample is a random vector in [-1,1]^d.
    The target is a smooth non‑linear function of the sum of the features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch ``Dataset`` that returns feature tensors and target scalars.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Fully‑connected “quantum” layer
# --------------------------------------------------------------------------- #
def FCL() -> nn.Module:
    """
    Return a lightweight module that mimics a parameterised quantum circuit
    by applying a linear transform followed by a tanh non‑linearity.
    """
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()


# --------------------------------------------------------------------------- #
# Estimator utilities
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """
    Evaluate a neural network for a list of parameter sets and a list of observables.
    Observables are callables that map a model output tensor to a scalar.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], Any]],
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
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """
    Adds Gaussian shot noise to the deterministic predictions.
    """
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], Any]],
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


# --------------------------------------------------------------------------- #
# Main regression model
# --------------------------------------------------------------------------- #
class RegressionModel(nn.Module):
    """
    A feed‑forward network that optionally exposes a fully‑connected “quantum” layer.
    The architecture is intentionally simple to keep the focus on the estimator utilities.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.fcl = FCL()  # expose the quantum‑like layer for comparative experiments

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)

    def evaluate_with_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run the FCL layer on a sequence of parameters and return a scalar expectation.
        """
        return self.fcl.run(thetas)


__all__ = [
    "RegressionModel",
    "RegressionDataset",
    "generate_superposition_data",
    "FCL",
    "FastBaseEstimator",
    "FastEstimator",
]
