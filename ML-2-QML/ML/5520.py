import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections.abc import Iterable, Sequence
from typing import Callable, List

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data where the target is a smooth
    non‑linear function of a random linear combination of the input
    features.  The function is chosen to be differentiable so that a
    neural network can learn it."""
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns a 1‑channel image representation of each sample."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        feat = self.features[index]
        size = int(np.ceil(np.sqrt(len(feat))))
        padded = np.pad(feat, (0, size * size - len(feat)), mode="constant")
        image = padded.reshape(1, size, size).astype(np.float32)
        return {
            "states": torch.tensor(image, dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridRegressionModel(nn.Module):
    """Classical CNN + fully‑connected regression network.
    The architecture is inspired by the Quantum‑NAT encoder and the
    simple regression network from the original seed."""
    def __init__(self, num_features: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        dummy = torch.randn(
            1, 1, int(np.ceil(np.sqrt(num_features))), int(np.ceil(np.sqrt(num_features)))
        )
        feat_dim = self.features(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.features(state_batch)
        out = self.head(features)
        return self.norm(out).squeeze(-1)

class FastEstimator:
    """Utility that evaluates a model on a list of parameter sets and
    optionally adds Gaussian shot noise."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
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
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = [[float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row] for row in results]
        return noisy

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data", "FastEstimator"]
