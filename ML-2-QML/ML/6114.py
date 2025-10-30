"""Enhanced classical feed‑forward regressor.

This module defines :class:`EstimatorQNN`, a PyTorch neural network
with batch‑normalisation, dropout and optional early‑stopping support.
It can be used directly as a drop‑in replacement for the original
tiny network while providing better generalisation on larger datasets.
"""

import torch
from torch import nn, optim
from torch.nn import functional as F
from typing import Optional, Tuple

class EstimatorQNN(nn.Module):
    """A flexible regression network.

    Parameters
    ----------
    input_dim : int, optional
        Dimensionality of the input feature vector.
        Defaults to 2 to keep compatibility with the original seed.
    hidden_dims : Sequence[int], optional
        Sizes of hidden layers. Defaults to ``(64, 32)``.
    dropout : float, optional
        Drop‑out probability applied after each hidden layer.
        Default ``0.1``.
    use_batchnorm : bool, optional
        Whether to apply batch‑normalisation after each linear layer.
        Default ``True``.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Tuple[int,...] = (64, 32),
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

    @staticmethod
    def train_loop(
        model: "EstimatorQNN",
        dataloader,
        criterion: nn.Module = nn.MSELoss(),
        optimizer: Optional[optim.Optimizer] = None,
        epochs: int = 200,
        device: str = "cpu",
        early_stopping_patience: int = 20,
    ) -> "EstimatorQNN":
        """A minimal training loop for quick prototyping.

        Parameters
        ----------
        model : EstimatorQNN
            The model to train.
        dataloader : Iterable
            Iterable yielding ``(inputs, targets)``.
        criterion : nn.Module, optional
            Loss function. Defaults to MSELoss.
        optimizer : optim.Optimizer, optional
            Optimiser. If ``None`` a ``Adam`` optimiser with
            ``lr=1e-3`` is created.
        epochs : int, optional
            Number of epochs to run.
        device : str, optional
            ``"cpu"`` or ``"cuda"``.
        early_stopping_patience : int, optional
            Number of epochs with no improvement before stopping.
        """
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.to(device)
        best_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataloader.dataset)

            # Early stopping
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    break
        return model

__all__ = ["EstimatorQNN"]
