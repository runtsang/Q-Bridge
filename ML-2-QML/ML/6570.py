import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data that mimics a superposition of |0...0> and |1...1>.
    The labels are a smooth function of the summed feature angles.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns a dict with'states' (float tensor) and 'target' (float tensor).
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ClassicalMLP(nn.Module):
    """
    Deep MLP that processes the raw feature vector before it is fed to the quantum encoder.
    """
    def __init__(self, num_features: int, hidden_dims: tuple[int,...] = (64, 32, 16)):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # 1‑dim output for regression
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class UnifiedQuantumRegressionModel(nn.Module):
    """
    Combines a classical MLP with a quantum encoder and a hybrid variational head.
    The quantum part can be toggled via the `quantum` flag, allowing ablation studies.
    """
    def __init__(self, num_features: int, quantum: bool = False):
        super().__init__()
        self.classical = ClassicalMLP(num_features)
        self.quantum = quantum
        if quantum:
            # In this lightweight classical version we ignore the quantum head
            pass

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        return self.classical(state_batch)

__all__ = ["RegressionDataset", "generate_superposition_data", "UnifiedQuantumRegressionModel"]
