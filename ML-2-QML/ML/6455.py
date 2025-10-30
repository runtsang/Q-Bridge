import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple

def generate_superposition_data(
    num_features: int,
    samples: int,
    num_wires: int,
    *,
    noise_std: float = 0.05,
    mix_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data that contains both a classical feature vector
    and a corresponding quantum state vector.

    Parameters
    ----------
    num_features : int
        Dimension of the classical feature vector.
    samples : int
        Number of samples to generate.
    num_wires : int
        Number of qubits used to construct the quantum state.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the labels.
    mix_ratio : float, optional
        Fraction of the label that comes from the quantum part
        (the rest comes from the classical part).

    Returns
    -------
    features : np.ndarray of shape (samples, num_features)
        Real‑valued classical features.
    states : np.ndarray of shape (samples, 2 ** num_wires)
        Complex quantum state vectors in computational basis.
    labels : np.ndarray of shape (samples,)
        Regression targets.
    """
    # Classical part
    features = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    classical_angles = features.sum(axis=1)
    classical_labels = np.sin(classical_angles)

    # Quantum part
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1

    quantum_labels = np.sin(2 * thetas) * np.cos(phis)

    # Mix
    labels = mix_ratio * quantum_labels + (1 - mix_ratio) * classical_labels
    labels += np.random.normal(scale=noise_std, size=labels.shape)

    return features, states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary containing real features,
    complex quantum states and the target value.
    """
    def __init__(self, samples: int, num_features: int, num_wires: int, mix_ratio: float = 0.5):
        self.features, self.states, self.labels = generate_superposition_data(
            num_features, samples, num_wires, mix_ratio=mix_ratio
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    A purely classical feed‑forward network that consumes both the
    real feature vector and the real/imaginary parts of the quantum
    state.  The architecture is intentionally deeper than the seed
    implementation to allow richer feature extraction.
    """
    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        input_dim = num_features + 2 * (2 ** num_wires)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # Concatenate real and imaginary parts of the quantum state
        state_real = states.real
        state_imag = states.imag
        x = torch.cat([features, state_real, state_imag], dim=-1)
        return self.net(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
