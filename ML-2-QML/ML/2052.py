"""EstimatorQNN – a small yet robust classical regressor.

This module replaces the one‑liner seed with a richer feed‑forward
network that includes batch‑normalisation, dropout and a convenient
training helper.  It keeps the same public API – a function
`EstimatorQNN()` that returns an instantiated :class:`torch.nn.Module`
which can be used directly in any PyTorch training pipeline.

The implementation is deliberately lightweight so it can be dropped
into notebooks or larger projects without external dependencies beyond
PyTorch.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def EstimatorQNN() -> nn.Module:
    """
    Return an instance of a fully‑connected regression network.

    The architecture is:
        2 → 32 → 16 → 1
    with batch‑normalisation, ReLU activations, and dropout after each
    hidden layer.  This gives a model that is both expressive enough for
    small regression tasks and regularised to avoid over‑fitting.

    Returns
    -------
    nn.Module
        A ready‑to‑train PyTorch model.
    """

    class _EstimatorNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.25),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.25),
                nn.Linear(16, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

        def predict(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
            """
            Run a forward pass on raw data and return the predictions.

            Parameters
            ----------
            X
                Input features of shape (n_samples, 2).

            Returns
            -------
            torch.Tensor
                Predicted values of shape (n_samples, 1).
            """
            device = next(self.parameters()).device
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float().to(device)
            with torch.no_grad():
                return self.forward(X)

        def train_on(
            self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            epochs: int = 100,
            batch_size: int = 32,
            lr: float = 1e-3,
            verbose: bool = False,
        ) -> None:
            """
            Very small convenience training loop.

            Parameters
            ----------
            X, y
                Training data.
            epochs, batch_size, lr
                Standard training hyper‑parameters.
            verbose
                If True, prints loss per epoch.
            """
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            dataset = TensorDataset(
                torch.from_numpy(X).float(),
                torch.from_numpy(y).float().unsqueeze(1),
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

            for epoch in range(epochs):
                epoch_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    pred = self.forward(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)
                epoch_loss /= len(dataset)
                if verbose:
                    print(f"Epoch {epoch + 1:03d} – loss: {epoch_loss:.6f}")

    return _EstimatorNet()
__all__ = ["EstimatorQNN"]
