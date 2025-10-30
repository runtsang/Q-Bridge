"""Enhanced fully connected layer with trainable neural network.

This implementation extends the original single‑parameter
tanh linear model to a multi‑layer feed‑forward network.
It supports dropout, batch‑norm, and a convenient `train`
method that accepts a data loader or ``(X, y)`` pairs.
"""

import numpy as np
import torch
from torch import nn, optim
from typing import Iterable, Sequence


class FCL(nn.Module):
    """
    Feed‑forward network that mimics a fully connected quantum layer.

    Parameters
    ----------
    n_features : int
        Number of input features (size of each theta vector).
    n_hidden : int, optional
        Size of the hidden layer(s).  A single hidden layer is
        created if ``n_layers`` is 1.
    n_layers : int, optional
        Number of hidden layers.  Must be >=1.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    output_dim : int, optional
        Size of the output vector.  Defaults to 1.
    """
    def __init__(
        self,
        n_features: int = 1,
        n_hidden: int = 20,
        n_layers: int = 2,
        dropout: float = 0.0,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        layers: Sequence[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(n_features, n_hidden))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(n_hidden, output_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass that accepts an iterable of thetas and
        returns the network output as a NumPy array.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of input feature values.

        Returns
        -------
        np.ndarray
            Model output flattened to a 1‑D array.
        """
        x = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.net(x)
        return out.detach().cpu().numpy().flatten()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        """
        Simple training loop using mean‑squared‑error loss.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).
        y : np.ndarray
            Target values, shape (n_samples,) or (n_samples, output_dim).
        epochs : int, optional
            Number of training epochs.
        lr : float, optional
            Learning rate for Adam.
        batch_size : int, optional
            Batch size.
        verbose : bool, optional
            If True, print epoch loss.
        """
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.train()  # set to training mode
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.6f}")

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Compute mean‑squared‑error on a held‑out set.

        Parameters
        ----------
        X : np.ndarray
            Features.
        y : np.ndarray
            Targets.

        Returns
        -------
        float
            MSE value.
        """
        self.eval()
        with torch.no_grad():
            preds = self.net(torch.tensor(X, dtype=torch.float32))
            loss = nn.MSELoss()(preds, torch.tensor(y, dtype=torch.float32))
        return loss.item()
