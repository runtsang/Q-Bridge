import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

def generate_superposition_data(num_features: int, samples: int, augment: bool = False, aug_factor: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data that mirrors the quantum superposition example but with optional
    data augmentation.  If ``augment`` is True, each original sample is
    replicated ``aug_factor`` times with small Gaussian noise added to the
    feature vector.  The label is computed from the sum of the features in the
    same way as the seed.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)

    if augment:
        noise = np.random.normal(scale=0.05, size=(samples * aug_factor, num_features)).astype(np.float32)
        x_aug = np.repeat(x, aug_factor, axis=0) + noise
        angles_aug = x_aug.sum(axis=1)
        y_aug = np.sin(angles_aug) + 0.1 * np.cos(2 * angles_aug)
        return x_aug, y_aug.astype(np.float32)

    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary with keys ``states`` (features) and
    ``target`` (label).  Handles both the vanilla and the augmented data
    formats produced by ``generate_superposition_data``.
    """
    def __init__(self, samples: int, num_features: int, augment: bool = False, aug_factor: int = 2):
        self.features, self.labels = generate_superposition_data(num_features, samples, augment, aug_factor)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegressionHybrid(nn.Module):
    """
    Classical feed‑forward network that first standardises the input, then
    passes it through a small MLP, followed by a dropout layer and a final
    linear head.  The network is deliberately deeper than the seed to
    demonstrate an extension while keeping the API identical.
    """
    def __init__(self, num_features: int, hidden_dims: tuple[int,...] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.scaler = StandardScaler()

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Standardise on the fly if the scaler has been fitted
        if hasattr(self.scaler,'mean_'):
            mean = torch.tensor(self.scaler.mean_, dtype=state_batch.dtype, device=state_batch.device)
            var = torch.tensor(self.scaler.var_, dtype=state_batch.dtype, device=state_batch.device)
            state_batch = (state_batch - mean) / torch.sqrt(var + 1e-6)
        return self.net(state_batch).squeeze(-1)

    def fit_scaler(self, data_loader: DataLoader):
        """
        Fit the ``StandardScaler`` on the entire training set.  This method
        must be called before the first forward pass when using the module.
        """
        all_features = []
        for batch in data_loader:
            all_features.append(batch["states"])
        all_features = torch.cat(all_features, dim=0).cpu().numpy()
        self.scaler.fit(all_features)

__all__ = ["QuantumRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
