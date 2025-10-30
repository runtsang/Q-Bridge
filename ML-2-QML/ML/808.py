"""Enhanced classical fully‑connected layer with training support."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def FCL(n_features: int = 1, hidden_dims: Iterable[int] = (32, 16), dropout: float = 0.1) -> nn.Module:
    """
    Return a fully‑connected neural network with optional dropout and batch‑norm.
    The network can be trained with :meth:`fit` and evaluated with :meth:`run`.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_dims : Iterable[int]
        Sizes of hidden layers. Defaults to a two‑layer network.
    dropout : float
        Dropout probability applied after each hidden layer.

    Returns
    -------
    nn.Module
        An instance of :class:`FullyConnectedLayer`.
    """
    class FullyConnectedLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            in_dim = n_features
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.model = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """
            Evaluate the network on a single sample of parameters.
            The parameters are interpreted as a 1‑D input vector.

            Returns
            -------
            np.ndarray
                The mean activation of the final layer.
            """
            x = torch.as_tensor(list(thetas), dtype=torch.float32).view(1, -1)
            out = self.forward(x)
            return out.detach().cpu().numpy()

        def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 200,
            lr: float = 1e-3,
            batch_size: int = 32,
            verbose: bool = False,
        ) -> None:
            """
            Train the network using mean‑squared error loss.

            Parameters
            ----------
            X : np.ndarray
                Input features of shape (n_samples, n_features).
            y : np.ndarray
                Target values of shape (n_samples,).
            epochs : int
                Number of training epochs.
            lr : float
                Learning rate for the Adam optimiser.
            batch_size : int
                Size of mini‑batches.
            verbose : bool
                If True, print loss every 10 epochs.
            """
            dataset = TensorDataset(torch.from_numpy(X).float(),
                                    torch.from_numpy(y).float().unsqueeze(1))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.parameters(), lr=lr)

            for epoch in range(1, epochs + 1):
                epoch_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    preds = self.forward(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)
                epoch_loss /= len(loader.dataset)
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} – Loss: {epoch_loss:.6f}")

    return FullyConnectedLayer()
