import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int, seed: int | None = None, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset based on superposition states with optional Gaussian noise.
    Parameters
    ----------
    num_features : int
        Number of qubits / features.
    samples : int
        Number of samples.
    seed : int | None
        Random seed for reproducibility.
    noise_std : float
        Standard deviation of Gaussian noise added to the labels.
    Returns
    -------
    states : np.ndarray of shape (samples, 2**num_features)
        Complex amplitude vectors.
    labels : np.ndarray of shape (samples,)
        Regression targets.
    """
    rng = np.random.default_rng(seed)
    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2 ** num_features), dtype=complex)
    for i in range(samples):
        omega0 = np.zeros(2 ** num_features, dtype=complex)
        omega0[0] = 1.0
        omega1 = np.zeros(2 ** num_features, dtype=complex)
        omega1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    if noise_std > 0:
        labels += rng.normal(scale=noise_std, size=labels.shape)
    return states.astype(complex), labels.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch dataset returning complex quantum states and regression targets.
    """
    def __init__(self, samples: int, num_features: int, seed: int | None = None, noise_std: float = 0.0):
        self.states, self.labels = generate_superposition_data(num_features, samples, seed, noise_std)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumRegressionModel(nn.Module):
    """
    Classical regression model operating on real‑imag concatenated features of quantum states.
    Architecture: Residual network with batch‑norm, ReLU, and dropout.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        input_dim = 2 * (2 ** num_features)  # real + imag
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        real = state_batch.real
        imag = state_batch.imag
        features = torch.cat([real, imag], dim=1)
        out = self.backbone(features)
        return self.head(out).squeeze(-1)

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
