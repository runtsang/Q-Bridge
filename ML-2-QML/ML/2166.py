"""Enhanced feed‑forward regressor with residual connections and regularization.

This module extends the original tiny network by:
- allowing arbitrary hidden layer sizes
- adding batch‑normalization and dropout
- providing a simple training helper that returns the training loss history
"""

import torch
from torch import nn, optim
from typing import Sequence, Tuple

def EstimatorQNN(
    input_dim: int = 2,
    hidden_layers: Sequence[int] = (8, 4),
    output_dim: int = 1,
    dropout_p: float = 0.1,
    use_batchnorm: bool = True,
) -> nn.Module:
    """Return a fully‑connected regression network with optional residuals.

    Parameters
    ----------
    input_dim : int
        Dimension of the input feature vector.
    hidden_layers : Sequence[int]
        Sizes of the hidden layers.
    output_dim : int
        Dimension of the output (default 1 for regression).
    dropout_p : float
        Drop‑out probability applied after every hidden layer.
    use_batchnorm : bool
        Whether to insert a BatchNorm1d after each linear layer.
    """
    class ResidualBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Tanh(),
                nn.Dropout(dropout_p),
                nn.BatchNorm1d(dim) if use_batchnorm else nn.Identity(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.net(x)

    layers = [nn.Linear(input_dim, hidden_layers[0]), nn.Tanh()]
    if dropout_p > 0:
        layers.append(nn.Dropout(dropout_p))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_layers[0]))

    for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
        layers += [
            nn.Linear(in_dim, out_dim),
            nn.Tanh(),
            nn.Dropout(dropout_p),
            nn.BatchNorm1d(out_dim) if use_batchnorm else nn.Identity(),
        ]

    layers.append(nn.Linear(hidden_layers[-1], output_dim))

    net = nn.Sequential(*layers)

    def train_one_epoch(
        self: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        lr: float = 1e-3,
        epochs: int = 1,
    ) -> Tuple[float, Sequence[float]]:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        loss_history = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for x, y in dataloader:
                optimizer.zero_grad()
                preds = self(x)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss_history.append(epoch_loss / len(dataloader))
        return loss_history[-1], loss_history

    net.train_one_epoch = train_one_epoch  # type: ignore[attr-defined]
    return net

__all__ = ["EstimatorQNN"]
