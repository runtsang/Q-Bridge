import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class HybridRegressionDataset(Dataset):
    """Dataset that provides classical features, quantum states, and targets."""
    def __init__(self, samples: int, num_features: int, num_wires: int):
        self.features, self.labels = self._generate_features(samples, num_features)
        self.states, _ = self._generate_states(samples, num_wires)

    def _generate_features(self, samples: int, num_features: int):
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    def _generate_states(self, samples: int, num_wires: int):
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0
        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_wires), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        return states, np.sin(2 * thetas) * np.cos(phis)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridRegressionModel(nn.Module):
    """Classical regression model augmented with a sampler network."""
    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.classical_net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # Sampler network inspired by SamplerQNN
        self.sampler_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # Combine classical and sampler outputs
        self.combiner = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.num_wires = num_wires

    def forward(self, batch: dict) -> torch.Tensor:
        feats = batch["features"]
        states = batch["states"]
        out_class = self.classical_net(feats).squeeze(-1)
        # Reduce complex state to two real features
        sampler_input = torch.stack([states.real, states.imag], dim=-1).mean(dim=1)
        out_samp = self.sampler_net(sampler_input)
        combined = torch.stack([out_class, out_samp[:, 0], out_samp[:, 1]], dim=-1)
        return self.combiner(combined).squeeze(-1)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict using only the classical part."""
        return self.classical_net(features).squeeze(-1)

__all__ = ["HybridRegressionDataset", "HybridRegressionModel"]
