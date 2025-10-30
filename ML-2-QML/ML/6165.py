import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Import the quantum part
from.qml_module import HybridEstimator as QuantumHybridEstimator

def generate_superposition_data(num_wires: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for regression. The states are complex amplitude
    vectors of the form cos(theta)|0..0> + exp(i*phi) sin(theta)|1..1>.
    Targets are sin(2*theta) * cos(phi).
    """
    import numpy as np
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
    return torch.tensor(states, dtype=torch.cfloat), torch.tensor(labels, dtype=torch.float32)

class RegressionDataset(Dataset):
    """
    Dataset yielding complex state vectors and regression targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {"states": self.states[idx], "target": self.labels[idx]}

class HybridEstimator(nn.Module):
    """
    Classical‑plus‑quantum regression model.
    The classical branch processes the real and imaginary parts of the state
    vector, the quantum branch extracts measurement statistics from a
    parameter‑driven circuit, and a shared linear head produces the final
    scalar output.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        # Classical feature extractor
        self.classical_net = nn.Sequential(
            nn.Linear(2 * (2 ** num_wires), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # Quantum feature extractor (from QML module)
        self.quantum_module = QuantumHybridEstimator(num_wires)
        # Shared head
        self.head = nn.Linear(32 + num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Classical path: concatenate real and imag parts
        real = state_batch.real
        imag = state_batch.imag
        classical_features = torch.cat([real, imag], dim=1)
        classical_features = self.classical_net(classical_features)

        # Quantum path
        quantum_features = self.quantum_module(state_batch)  # shape (batch, num_wires)

        # Concatenate and predict
        features = torch.cat([classical_features, quantum_features], dim=1)
        return self.head(features).squeeze(-1)

__all__ = ["HybridEstimator", "RegressionDataset", "generate_superposition_data"]
