"""PyTorch regression model utilities for the ML baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _ensure_float_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a float32 tensor on CPU."""
    if isinstance(array, torch.Tensor):
        tensor = array
    else:
        tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class TorchMLPRegressor:
    """A simple 3-layer MLP regressor implemented with PyTorch."""

    n_features: int
    hidden_layer_sizes: Sequence[int] = (64, 32)
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 200
    batch_size: int = 32
    seed: int = 42
    device: torch.device | None = None
    history: List[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if len(self.hidden_layer_sizes) != 2:
            raise ValueError("hidden_layer_sizes must contain exactly two integers for a 3-layer MLP.")
        self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        layers: List[nn.Module] = []
        in_dim = self.n_features
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers).to(self.device)
        print(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_fn = nn.MSELoss()

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> "TorchMLPRegressor":
        """Train the regressor on the provided dataset."""
        X_tensor = _ensure_float_tensor(np.asarray(X)).to(self.device)
        y_tensor = _ensure_float_tensor(np.asarray(y)).view(-1, 1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model.train()
        self.history.clear()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad(set_to_none=True)
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            self.history.append(epoch_loss)
            if verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                print(f"    Epoch {epoch + 1:4d}/{self.epochs} - loss={epoch_loss:.6f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for the provided features."""
        X_tensor = _ensure_float_tensor(np.asarray(X)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().ravel()
        return preds


def build_mlp(
    n_features: int,
    hidden_layer_sizes: Tuple[int, int] = (64, 32),
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 200,
    batch_size: int = 32,
    seed: int = 42,
    device: torch.device | None = None,
) -> TorchMLPRegressor:
    """Factory that returns a :class:`TorchMLPRegressor` with the requested hyper-parameters."""

    return TorchMLPRegressor(
        n_features=n_features,
        hidden_layer_sizes=hidden_layer_sizes,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
