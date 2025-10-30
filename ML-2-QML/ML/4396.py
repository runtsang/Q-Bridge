import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Autoencoder import Autoencoder
from GraphQNN import fidelity_adjacency

class RegressionDataset(Dataset):
    """Dataset generating superposition‑based regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = self._generate_superposition_data(num_features, samples)

    @staticmethod
    def _generate_superposition_data(num_features: int, samples: int):
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridRegressionModel(nn.Module):
    """Classical hybrid regression model using an autoencoder, graph similarity, and a feedforward head."""
    def __init__(
        self,
        num_features: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        graph_threshold: float = 0.9,
        graph_secondary: float | None = None,
    ):
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim=num_features,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.graph_threshold = graph_threshold
        self.graph_secondary = graph_secondary

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder.encode(states)
        return self.head(latent).squeeze(-1)

    def compute_graph(self, latent: torch.Tensor) -> torch.Tensor:
        """Return weighted adjacency matrix from fidelity between latent vectors."""
        states = [latent[i] for i in range(latent.size(0))]
        graph = fidelity_adjacency(states, self.graph_threshold, secondary=self.graph_secondary)
        adjacency = torch.zeros((latent.size(0), latent.size(0)), dtype=torch.float32)
        for i, j, data in graph.edges(data=True):
            adjacency[i, j] = data.get("weight", 1.0)
        return adjacency

    def fit(self, dataset: Dataset, epochs: int = 20, lr: float = 1e-3, device: torch.device | None = None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for batch in loader:
                states = batch["states"].to(device)
                target = batch["target"].to(device)
                optimizer.zero_grad()
                pred = self(states)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(states)

__all__ = ["HybridRegressionModel", "RegressionDataset"]
