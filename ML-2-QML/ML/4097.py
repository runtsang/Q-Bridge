import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import List, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class SelfAttention(nn.Module):
    """
    Classical self‑attention block that mimics the quantum interface.
    Parameters are expected in the same shape as used in the quantum seed:
    rotation_params: (..., 3*embed_dim)
    entangle_params: (..., embed_dim-1)
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, rotation_params: torch.Tensor,
                entangle_params: torch.Tensor,
                inputs: torch.Tensor) -> torch.Tensor:
        query = torch.matmul(inputs, rotation_params.view(-1, self.embed_dim))
        key   = torch.matmul(inputs, entangle_params.view(-1, self.embed_dim))
        scores = torch.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

class FastBaseEstimator:
    """
    Lightweight estimator that can wrap any torch.nn.Module.
    It offers deterministic evaluation and optional shot‑style Gaussian noise.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
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
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
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

    def add_self_attention(self, embed_dim: int, position: int = -1) -> None:
        """
        Insert a SelfAttention layer into the model's architecture.
        The layer is appended after the last layer if ``position`` is -1,
        otherwise it is inserted at the given index.
        """
        attention = SelfAttention(embed_dim)
        if isinstance(self.model, nn.Sequential):
            layers = list(self.model)
            if position == -1:
                layers.append(attention)
            else:
                layers.insert(position, attention)
            self.model = nn.Sequential(*layers)
        else:
            raise TypeError("Self‑attention insertion currently supports nn.Sequential models only.")

# ------------------------------------------------------------------
# Regression utilities (classical)
# ------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    A small feed‑forward network that mirrors the quantum model's structure.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

__all__ = ["FastBaseEstimator", "SelfAttention", "RegressionDataset", "QModel", "generate_superposition_data"]
