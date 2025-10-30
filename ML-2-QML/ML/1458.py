"""Enhanced classical QCNN with regularisation and training helper.

The network mirrors the original structure but adds dropout, batch‑norm
and a lightweight early‑stopping training routine that can be used
directly on NumPy or Torch tensors.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from typing import Tuple, Optional

class QCNNHybrid(nn.Module):
    """Fully connected neural network mimicking a QCNN with regularisation."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Tuple[int,...] = (16, 16, 12, 8, 4, 4),
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim

        def add_block(out_dim: int) -> None:
            nonlocal prev_dim
            layers.append(nn.Linear(prev_dim, out_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = out_dim

        # Feature map
        add_block(hidden_dims[0])
        # Convolution / pooling stages
        for dim in hidden_dims[1:]:
            add_block(dim)

        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNHybrid:
    """Return a ready‑to‑train QCNNHybrid instance."""
    return QCNNHybrid()

def train(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 20,
) -> Tuple[nn.Module, list]:
    """Simple training loop with early stopping on validation loss."""
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze()
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb).squeeze()
                loss = criterion(logits, yb.float())
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        history.append((epoch_loss, val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history

__all__ = ["QCNNHybrid", "QCNN", "train"]
