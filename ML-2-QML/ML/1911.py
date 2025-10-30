import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data in the same style as the original seed.
    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the labels.
    Returns
    -------
    X : np.ndarray of shape (samples, num_features)
        Feature matrix.
    y : np.ndarray of shape (samples,)
        Target vector.
    """
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += noise_std * np.random.randn(samples)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Torch Dataset wrapping the synthetic superposition data.
    """
    def __init__(self, samples: int, num_features: int, device: torch.device = torch.device('cpu')):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.device = device

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32, device=self.device),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32, device=self.device)
        }

class QModel(nn.Module):
    """
    Classical regression model with optional dropout for uncertainty estimation.
    """
    def __init__(self, num_features: int, dropout_p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_with_dropout(self, x: torch.Tensor, n_samples: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate predictive uncertainty by performing multiple stochastic forward passes with dropout enabled.
        Parameters
        ----------
        x : torch.Tensor
            Input batch.
        n_samples : int
            Number of stochastic forward passes.
        Returns
        -------
        mean_pred : torch.Tensor
            Mean prediction over stochastic passes.
        std_pred : torch.Tensor
            Standard deviation of predictions (uncertainty estimate).
        """
        self.train()
        preds = []
        for _ in range(n_samples):
            preds.append(self.forward(x))
        preds = torch.stack(preds, dim=0)
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0, unbiased=False)
        self.eval()
        return mean_pred, std_pred

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
