import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Iterable, List

# Dataset utilities (QuantumRegression)
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int) -> None:
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# Classical FCL layer (from FCL.py)
class FullyConnectedLayer(nn.Module):
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

# Sampler network (from SamplerQNN.py)
class SamplerModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return nn.functional.softmax(self.net(inputs), dim=-1)

# Hybrid quantum‑classical regression model
class HybridQuantumRegression(nn.Module):
    def __init__(self, num_features: int, num_qubits: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.encoder = nn.Sequential(
            nn.Linear(num_features, num_qubits * 2),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(num_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, estimator: callable | None = None) -> torch.Tensor:
        params = self.encoder(x).view(-1, self.num_qubits, 2)
        if estimator is None:
            return params
        expectations = estimator(params)
        return self.head(expectations)

# Utility to evaluate a quantum circuit
def evaluate_quantum(params: torch.Tensor, circuit) -> torch.Tensor:
    """
    Evaluate the quantum circuit for a batch of parameters.

    Parameters
    ----------
    params : torch.Tensor
        Tensor of shape (batch, num_qubits, 2) containing rotation angles.
    circuit : object
        Quantum circuit object that implements a `run` method accepting a list of
        flat parameter lists.

    Returns
    -------
    torch.Tensor
        Expectation values of shape (batch, num_qubits).
    """
    param_list = params.cpu().numpy().reshape((params.shape[0], -1)).tolist()
    exp_vals = circuit.run(param_list)
    return torch.tensor(exp_vals, dtype=torch.float32)

__all__ = [
    "FullyConnectedLayer",
    "SamplerModule",
    "HybridQuantumRegression",
    "evaluate_quantum",
    "RegressionDataset",
    "generate_superposition_data",
]
