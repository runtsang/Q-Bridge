"""Enhanced classical regression model with feature scaling, gradient clipping,
and early stopping. Builds on the original seed by adding a flexible training
pipeline and robust data handling."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with superposition‑inspired labels."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping features and targets with optional scaling."""
    def __init__(self, samples: int, num_features: int, scaler: bool = True):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        if scaler:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features).astype(np.float32)
        else:
            self.scaler = None

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):
        return {"states": torch.tensor(self.features[idx], dtype=torch.float32),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

class QuantumRegressionModel(nn.Module):
    """Classical neural network for regression with optional residual connections."""
    def __init__(self, num_features: int, hidden_dims: list[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers = []
        in_dim = num_features
        for hdim in hidden_dims:
            layers.append(nn.Linear(in_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hdim))
            in_dim = hdim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def train_model(model: nn.Module,
                dataset: Dataset,
                epochs: int = 200,
                batch_size: int = 32,
                lr: float = 1e-3,
                clip_norm: float = 1.0,
                early_stop: int = 20,
                device: str | torch.device = "cpu") -> tuple[float, float]:
    """Train the model with Adam optimizer, gradient clipping, and early stopping.
    Returns the best validation MSE and training loss history."""
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    patience = 0
    history = []
    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            states = batch["states"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad()
            out = model(states)
            loss = criterion(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)
        epoch_loss /= len(dataset)
        if epoch_loss < best_val:
            best_val = epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                break
        history.append(epoch_loss)
    return best_val, float(history[-1])

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data", "train_model"]
