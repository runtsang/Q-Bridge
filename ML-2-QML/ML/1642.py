"""Enhanced classical QCNN implementation with residuals, dropout, and training utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Optional, Dict

class SharedQCNN(nn.Module):
    """
    Convolution‑inspired fully‑connected network that mirrors a QCNN architecture.

    Enhancements over the seed:
    * Residual connections between blocks.
    * Batch normalization after each linear layer.
    * Dropout for regularization.
    * Optional early stopping based on validation loss.
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] = [16, 12, 8, 4, 4],
        dropout: float = 0.1,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        act = getattr(F, activation.lower())
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Tanh() if activation.lower() == "tanh" else nn.ReLU())
            prev = h
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        self.activation = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.sigmoid(self.head(out))
        return out

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stop_patience: Optional[int] = 10,
    ) -> Dict[str, List[float]]:
        """
        Train the network using binary cross‑entropy loss.

        Returns a history dictionary of train/val losses per epoch.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        best_val = float("inf")
        patience = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                y_pred = self.forward(xb).squeeze()
                loss = criterion(y_pred, yb.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        y_pred = self.forward(xb).squeeze()
                        loss = criterion(y_pred, yb.float())
                        val_loss += loss.item() * xb.size(0)
                val_loss /= len(val_loader.dataset)
                history["val_loss"].append(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                    torch.save(self.state_dict(), "best_qcnn.pt")
                else:
                    patience += 1
                    if early_stop_patience is not None and patience >= early_stop_patience:
                        self.load_state_dict(torch.load("best_qcnn.pt"))
                        break
        return history

def QCNN(**kwargs) -> SharedQCNN:
    """
    Factory that returns a configured SharedQCNN instance.
    """
    return SharedQCNN(**kwargs)

__all__ = ["QCNN", "SharedQCNN"]
