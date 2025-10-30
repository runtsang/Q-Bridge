from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states of the form cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩ and a regression target.

    Parameters
    ----------
    num_wires : int
        Number of qubits / input feature dimension.
    samples : int
        Number of samples to generate.

    Returns
    -------
    states : ndarray
        Complex amplitude vectors of shape (samples, 2**num_wires).
    labels : ndarray
        Continuous targets of shape (samples,).
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that can return either classical features or quantum‑state representations.

    Parameters
    ----------
    samples : int
        Number of samples in the dataset.
    num_features : int
        Feature dimension / number of qubits.
    use_quantum : bool, default False
        If ``True`` the dataset returns a complex state vector per sample; otherwise it
        returns a real feature vector.
    """
    def __init__(self, samples: int, num_features: int, use_quantum: bool = False):
        if use_quantum:
            states, labels = generate_superposition_data(num_features, samples)
            self.features = torch.tensor(states, dtype=torch.cfloat)
        else:
            x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
            angles = x.sum(axis=1)
            labels = np.sin(angles) + 0.1 * np.cos(2 * angles)
            self.features = torch.tensor(x, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {"features": self.features[index], "target": self.labels[index]}


class QuantumFeatureEncoder(nn.Module):
    """Simple quantum‑inspired feature map that returns the expectation value of Z
    after an RX rotation on the |0⟩ state.

    For an input vector ``x`` the output is ``cos(x)``, which is the analytic
    expectation value ⟨0|RX(x)† Z RX(x) |0⟩.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.cos(x)


class HybridRegressionModel(nn.Module):
    """A hybrid neural network that optionally augments its input with quantum‑inspired
    features before passing it through a classical feed‑forward network.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input / number of qubits.
    hidden_sizes : tuple[int,...], default (64, 32)
        Sizes of the hidden layers.
    depth : int, default 1
        Depth of the quantum feature encoder (currently unused but kept for API
        compatibility with the quantum counterpart).
    use_quantum : bool, default True
        If ``True`` the input is first transformed by ``QuantumFeatureEncoder``.
    """
    def __init__(
        self,
        num_features: int,
        hidden_sizes: tuple[int,...] = (64, 32),
        depth: int = 1,
        use_quantum: bool = True,
    ):
        super().__init__()
        self.use_quantum = use_quantum
        self.encoder = QuantumFeatureEncoder() if use_quantum else nn.Identity()

        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            x = self.encoder(x)
        return self.net(x).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
