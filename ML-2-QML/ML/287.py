\
"""Enhanced classical regression with cross‑validation, data augmentation, and configurable MLP."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    noise_level: float = 0.0,
    augment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics a quantum superposition.
    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples.
    noise_level : float, optional
        Standard deviation of Gaussian noise added to the target.
    augment : bool, optional
        If True, apply a random orthogonal rotation to the feature vectors.
    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Target vector of shape (samples,).
    """
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if augment:
        # Random orthogonal rotation
        Q, _ = np.linalg.qr(np.random.randn(num_features, num_features))
        X = X @ Q
    if noise_level > 0.0:
        y += np.random.normal(0.0, noise_level, size=y.shape)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset wrapper that returns a dictionary of tensors.
    """
    def __init__(self, samples: int, num_features: int, *, augment: bool = False):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, augment=augment
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Configurable MLP with optional residual shortcut.
    """
    def __init__(self, num_features: int, hidden_sizes: list[int] | None = None):
        super().__init__()
        hidden_sizes = hidden_sizes or [32, 16, 8]
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.residual = nn.Identity()

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        out = self.net(state_batch)
        return self.residual(out).squeeze(-1)

def cross_validate(
    model_cls: type[QModel],
    X: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 50,
    device: torch.device | None = None,
    seed: int = 42,
    hidden_sizes: list[int] | None = None,
) -> dict[str, float]:
    """
    Perform k‑fold cross‑validation on a given model class.
    Returns the average MSE over the validation folds.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    mse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        val_ds = DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            ),
            batch_size=batch_size,
        )

        model = model_cls(X.shape[1], hidden_sizes=hidden_sizes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience = 5
        counter = 0

        for epoch in range(epochs):
            model.train()
            for batch in train_ds:
                optimizer.zero_grad()
                pred = model(batch[0].to(device))
                loss = criterion(pred, batch[1].to(device))
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch in val_ds:
                    pred = model(batch[0].to(device))
                    val_preds.append(pred.cpu())
                    val_targets.append(batch[1])
            val_pred = torch.cat(val_preds).numpy()
            val_true = torch.cat(val_targets).numpy()
            val_loss = mean_squared_error(val_true, val_pred)
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        mse_scores.append(best_loss)

    return {"mse": np.mean(mse_scores)}

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data", "cross_validate"]
