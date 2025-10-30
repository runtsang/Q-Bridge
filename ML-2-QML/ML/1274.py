import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate random superposition‑like data for regression."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

class FeatureMapper(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class QModel(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 32):
        super().__init__()
        self.mapper = FeatureMapper(num_features, hidden_dim)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.skip = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        mapped = self.mapper(state_batch)
        out = self.net(mapped)
        return out + self.skip(mapped)

    def fit(self, dataset: Dataset, epochs: int = 10, lr: float = 1e-3, device: str = "cpu"):
        self.to(device)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            for batch in loader:
                optimizer.zero_grad()
                preds = self.forward(batch["states"].to(device))
                loss = loss_fn(preds, batch["target"].to(device))
                loss.backward()
                optimizer.step()
        self.eval()
