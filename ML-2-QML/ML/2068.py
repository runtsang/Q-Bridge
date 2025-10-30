"""Enhanced classical regression model with residual connections and dropout.

The model builds on the original seed by adding a configurable deep neural
network that can adapt its depth, width, and dropout rate.  It also
exposes a lightweight ``predict`` method for inference and a
``train_with_early_stopping`` helper that accepts an optional
``patience`` parameter.  This extension allows the same class to be
used in both research prototypes and production pipelines without
changing downstream code.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def generate_superposition_data(
    num_features: int,
    samples: int,
    noise_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where ``y`` is a
    non‑linear function of the summed feature vector.

    Parameters
    ----------
    num_features :
        Dimensionality of each sample.
    samples :
        Number of samples to generate.
    noise_std :
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    X, y :
        Features and targets as ``float32`` arrays.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += np.random.normal(0, noise_std, size=y.shape).astype(np.float32)
    return x, y


class RegressionDataset(Dataset):
    """
    PyTorch Dataset wrapping the synthetic regression data.

    Parameters
    ----------
    samples :
        Number of samples in the dataset.
    num_features :
        Dimensionality of each sample.
    """

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(
            num_features, samples
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """A simple residual block with two linear layers and a GELU activation."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class QModel(nn.Module):
    """
    A configurable deep neural network for regression with residual
    connections and dropout.

    Parameters
    ----------
    num_features :
        Input dimensionality.
    hidden_sizes :
        Sequence of hidden layer sizes.  The default creates a 4‑layer
        network with 128, 64, 32, and 16 neurons.
    dropout :
        Dropout probability applied after each residual block.
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: tuple[int,...] = (128, 64, 32, 16),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = num_features
        for size in hidden_sizes:
            layers.append(ResidualBlock(prev_dim, dropout=dropout))
            layers.append(nn.Linear(prev_dim, size))
            prev_dim = size
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

    def predict(
        self,
        X: torch.Tensor,
        batch_size: int = 256,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Run inference on a large batch of inputs.

        Parameters
        ----------
        X :
            Input tensor of shape ``(N, D)``.
        batch_size :
            Number of samples processed per GPU/CPU chunk.
        device :
            Target device.  If ``None`` the model's device is used.

        Returns
        -------
        torch.Tensor :
            Predicted values of shape ``(N,)``.
        """
        self.eval()
        device = device or next(self.parameters()).device
        X = X.to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                batch = X[i : i + batch_size]
                preds.append(self(batch))
        return torch.cat(preds)

    def train_with_early_stopping(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 100,
        patience: int = 10,
        device: torch.device | str | None = None,
    ) -> list[float]:
        """
        Train the model with optional early stopping.

        Parameters
        ----------
        train_loader :
            DataLoader for training data.
        val_loader :
            DataLoader for validation data.
        loss_fn :
            Loss function (e.g., nn.MSELoss()).
        optimizer :
            Optimizer instance.
        epochs :
            Maximum number of training epochs.
        patience :
            Number of epochs with no improvement before stopping.
        device :
            Target device.  If ``None`` the model's device is used.

        Returns
        -------
        List[float] :
            Validation loss history.
        """
        self.to(device or "cpu")
        best_loss = float("inf")
        best_state = None
        history = []

        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                y_hat = self(batch["states"])
                loss = loss_fn(y_hat, batch["target"])
                loss.backward()
                optimizer.step()

            # Validation step
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    y_hat = self(batch["states"])
                    val_loss += loss_fn(y_hat, batch["target"]).item() * len(batch["target"])
            val_loss /= len(val_loader.dataset)
            history.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
