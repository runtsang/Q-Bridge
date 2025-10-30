"""Enhanced classical feed‑forward regressor with configurable architecture and training utilities."""

from __future__ import annotations
from typing import Sequence, Callable, Iterable

import torch
from torch import nn, optim

class EstimatorQNNGen(nn.Module):
    """A flexible fully‑connected neural network for regression.

    Parameters
    ----------
    input_dim : int
        Size of each input sample.
    hidden_dims : Sequence[int], optional
        Number of units in each hidden layer. Defaults to ``[8, 4]``.
    activation : Callable[[torch.Tensor], torch.Tensor], optional
        Non‑linearity applied after each hidden layer. Defaults to ``nn.Tanh``.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 4]
        if activation is None:
            activation = nn.Tanh()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_optimizer(model: nn.Module, lr: float = 1e-3) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=lr)

    @staticmethod
    def _default_loss() -> nn.Module:
        return nn.MSELoss()

    def train_on_loader(
        self,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 10,
        lr: float = 1e-3,
        loss_fn: nn.Module | None = None,
        device: torch.device | str | None = None,
    ) -> list[float]:
        """Train the network using the provided data loader.

        Parameters
        ----------
        loader
            Iterable yielding ``(inputs, targets)`` tuples.
        epochs
            Number of training epochs.
        lr
            Learning rate for Adam.
        loss_fn
            Optional loss function; defaults to MSE.
        device
            Device to run the training on. If ``None``, uses ``torch.device('cpu')``.
        Returns
        -------
        losses
            List of epoch‑wise training losses.
        """
        device = device or torch.device("cpu")
        self.to(device)
        self.train()
        loss_fn = loss_fn or self._default_loss()
        optimizer = self._default_optimizer(self, lr)
        epoch_losses: list[float] = []

        for _ in range(epochs):
            batch_losses: list[float] = []
            for inp, tgt in loader:
                inp, tgt = inp.to(device), tgt.to(device)
                optimizer.zero_grad()
                pred = self(inp)
                loss = loss_fn(pred, tgt)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(sum(batch_losses) / len(batch_losses))
        return epoch_losses

    def predict(self, x: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
        """Return predictions for ``x``."""
        device = device or torch.device("cpu")
        self.eval()
        with torch.no_grad():
            return self(x.to(device))
