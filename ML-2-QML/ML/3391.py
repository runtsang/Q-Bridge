import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int):
    """Generate synthetic regression data based on sinusoidal superposition of features."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridFCL(nn.Module):
    """Classical fully‑connected network that can act as a stand‑in for the quantum layer.
    It implements the same ``run`` interface as the original FCL example but uses a small
    feed‑forward architecture instead of a single linear layer."""
    def __init__(self, n_features: int = 1, hidden_sizes=(32, 16)):
        super().__init__()
        layers = []
        input_size = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            input_size = h
        layers.append(nn.Linear(input_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def run(self, thetas: list[float]) -> np.ndarray:
        """Mimic the original ``FCL.run`` interface for compatibility.
        Thetas are interpreted as linear weights applied to a single‑feature input."""
        values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        # Use the first linear layer of the network to approximate
        # the behaviour of the original FCL.
        linear = self.net[0]
        out = torch.tanh(linear(values)).mean(dim=0)
        return out.detach().numpy()
