"""
Hybrid classical regression module that mirrors the quantum regression
example while enabling efficient evaluation of many parameter sets.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Sequence, Iterable, List, Callable

# ----------------------------------------------------
# Dataset utilities
# ----------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data where the target is a non‑linear
    function of the sum of the input features.  This matches the
    behaviour of the quantum data generator but is purely classical.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset returning feature vectors and scalar targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ----------------------------------------------------
# Self‑attention helper (classical)
# ----------------------------------------------------
def SelfAttention():
    class ClassicalSelfAttention:
        """Simple self‑attention layer implemented with PyTorch."""
        def __init__(self, embed_dim: int):
            self.embed_dim = embed_dim

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
        ) -> np.ndarray:
            query = torch.as_tensor(
                inputs @ rotation_params.reshape(self.embed_dim, -1),
                dtype=torch.float32,
            )
            key = torch.as_tensor(
                inputs @ entangle_params.reshape(self.embed_dim, -1),
                dtype=torch.float32,
            )
            value = torch.as_tensor(inputs, dtype=torch.float32)
            scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
            return (scores @ value).numpy()

    return ClassicalSelfAttention(embed_dim=4)

# ----------------------------------------------------
# Estimator utilities
# ----------------------------------------------------
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Guarantee a 2‑D tensor for a single parameter set."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """
    Evaluate a PyTorch model for many parameter sets and a list of
    observable functions.  Inspired by the lightweight estimator
    in the original repo.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
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
    """
    Same as FastBaseEstimator but adds optional Gaussian shot noise
    to mimic measurement statistics.
    """
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
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

# ----------------------------------------------------
# Classical regression model
# ----------------------------------------------------
class HybridRegression(nn.Module):
    """
    A lightweight MLP that optionally augments its input with a
    classical self‑attention block.  The model is deliberately simple
    so that it can act as a proxy for a quantum circuit when used
    with the estimator utilities.
    """
    def __init__(
        self,
        num_features: int,
        attention: bool = False,
        attention_dim: int = 4,
    ):
        super().__init__()
        self.attention = attention
        if attention:
            self.attn = SelfAttention()()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attention:
            # expand 1‑D inputs to 2‑D for attention
            batch = x.shape[0]
            x_2d = x.reshape(batch, -1)
            # Random parameters for the attention block
            rot = np.random.rand(self.attention_dim, x_2d.shape[1])
            ent = np.random.rand(self.attention_dim, x_2d.shape[1])
            attn_out = self.attn.run(rot, ent, x_2d.numpy())
            x = torch.tensor(attn_out, dtype=x.dtype, device=x.device)
        return self.net(x).squeeze(-1)

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "SelfAttention",
    "FastBaseEstimator",
    "FastEstimator",
    "HybridRegression",
]
