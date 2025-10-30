"""Hybrid fully‑connected layer with classical and quantum backends.

The class mirrors the API of the original FCL seed but extends it with a
trainable dense network and a small helper dataset that emulates the
superposition data used in the quantum regression example.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

__all__ = ["HybridFCL", "RegressionDataset", "generate_superposition_data"]

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce the same toy data used in the quantum regression seed.

    Parameters
    ----------
    num_features : int
        Feature dimension of each sample.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Feature matrix X and target vector y.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Torch dataset that packages the superposition data for quick training.
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

class HybridFCL(nn.Module):
    """
    Classical implementation of a fully‑connected layer that supports
    arbitrary feature dimension and a small hidden network.
    """
    def __init__(self, n_features: int = 1, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward pass that applies the linear network to a 1‑D sequence of
        parameters and returns the mean prediction as a numpy array.

        Parameters
        ----------
        thetas : Iterable[float]
            Sequence of real parameters.

        Returns
        -------
        np.ndarray
            1‑D array of predictions.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        prediction = self.net(values).mean(dim=0)
        return prediction.detach().cpu().numpy()

    def fit(self, dataset: Dataset, epochs: int = 10, lr: float = 0.01) -> None:
        """
        Quick training loop using mean‑squared error loss.

        Parameters
        ----------
        dataset : Dataset
            Dataset providing ``states`` and ``target``.
        epochs : int, optional
            Number of training epochs.
        lr : float, optional
            Learning rate for the Adam optimiser.
        """
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            for batch in loader:
                optimizer.zero_grad()
                preds = self.net(batch["states"])
                loss = criterion(preds.squeeze(-1), batch["target"])
                loss.backward()
                optimizer.step()
