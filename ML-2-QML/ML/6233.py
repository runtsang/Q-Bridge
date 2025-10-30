"""Enhanced classical regressor with modular layers and a training helper."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


class EstimatorNN(nn.Module):
    """Fully‑connected regression network with bias, dropout and a tunable activation.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input features.
    hidden_dims : Sequence[int], default (8, 4)
        Sizes of hidden layers.
    output_dim : int, default 1
        Dimensionality of the output.
    activation : nn.Module, default nn.Tanh
        Activation function applied after every hidden layer.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int,...] = (8, 4),
        output_dim: int = 1,
        activation: nn.Module = nn.Tanh,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def EstimatorQNN(**kwargs) -> EstimatorNN:
    """Convenience factory returning a configured EstimatorNN."""
    return EstimatorNN(**kwargs)


def train_estimator(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cpu",
) -> nn.Module:
    """Train the model with Adam, ReduceLROnPlateau and early stopping.

    Parameters
    ----------
    model : nn.Module
        The network to train.
    dataloader : DataLoader
        Iterable over (inputs, targets) pairs.
    epochs : int
        Maximum number of epochs.
    lr : float
        Initial learning rate.
    patience : int
        Number of epochs with no improvement before stopping.
    device : str
        Device where training takes place.

    Returns
    -------
    nn.Module
        The best‑performing model snapshot.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, verbose=True
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(dataloader.dataset)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping after epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


__all__ = ["EstimatorNN", "EstimatorQNN", "train_estimator"]
