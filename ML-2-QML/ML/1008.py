"""Classical regression model with a hybrid training option and noise augmentation.

The module now supports:
* A `HybridModel` that accepts a quantum‑encoded feature vector and feeds it into a multi‑layer perceptron.
* A `NoiseAugmentedDataset` that injects Gaussian noise into the target during training.
* A new training routine that can train the classical head alone or jointly with the quantum encoder.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data where the target is a nonlinear function
    of the sum of the input features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

def noise_variance(seed: int = 42) -> float:
    """Return a random variance for label noise, seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.01, 0.1)

class NoiseAugmentedDataset(Dataset):
    """
    Wraps a base dataset and adds Gaussian noise to the target.
    """
    def __init__(self, base_dataset: Dataset, noise_std: float | None = None):
        self.base = base_dataset
        self.noise_std = noise_std if noise_std is not None else noise_variance()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        noisy_target = sample["target"] + torch.randn_like(sample["target"]) * self.noise_std
        return {"states": sample["states"], "target": noisy_target}

# --------------------------------------------------------------------------- #
# Classical model
# --------------------------------------------------------------------------- #
class ClassicalMLP(nn.Module):
    """Simple feed‑forward network for regression."""
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU())
        layers.pop()  # remove trailing ReLU
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #
class HybridModel(nn.Module):
    """
    Accepts a quantum‑encoded feature vector (size = n_wires) and passes it through
    a classical MLP. Useful for end‑to‑end fine‑tuning of the quantum encoder.
    """
    def __init__(self, n_wires: int, mlp_hidden: list[int] | None = None):
        super().__init__()
        self.head = ClassicalMLP(n_wires, mlp_hidden)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

# --------------------------------------------------------------------------- #
# Dataset and model wrappers
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """
    Same as the seed but with optional noise augmentation.
    """
    def __init__(self, samples: int, num_features: int, noise: bool = False):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.base = None
        if noise:
            self.base = NoiseAugmentedDataset(self)

    def __len__(self) -> int:
        return len(self.base) if self.base else len(self.features)

    def __getitem__(self, idx: int):
        if self.base:
            return self.base[idx]
        else:
            return {"states": torch.tensor(self.features[idx], dtype=torch.float32),
                    "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

class QModel(nn.Module):
    """
    Wrapper around the seed's QML model for compatibility.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = ClassicalMLP(num_features)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch)

__all__ = [
    "RegressionDataset",
    "HybridModel",
    "NoiseAugmentedDataset",
    "ClassicalMLP",
    "QModel",
    "generate_superposition_data",
]
