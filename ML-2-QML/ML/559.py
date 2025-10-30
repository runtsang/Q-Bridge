"""QCNNHybridModel – a classical, convolution‑inspired neural network with hyper‑parameter search.

The model mirrors the original QCNN architecture but adds:
* Configurable number of convolution‑pool blocks.
* Optional dropout after each block.
* Early‑stopping based on validation loss.
* Grid‑search support for learning rate, dropout, and hidden layer sizes.
* Optional ONNX export for deployment.

The public API matches scikit‑learn estimators (fit, predict, score) for easy integration.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from typing import Iterable, Tuple, Any, Dict, Iterable


class QCNNHybridModel(nn.Module):
    """
    Classic QCNN‑inspired architecture with optional dropout and configurable depth.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features (default 8).
    hidden_dims : Iterable[int] | None
        Sequence of hidden layer sizes for each convolutional block.
    dropout : float
        Dropout probability applied after each block.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Iterable[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 8, 4]
        self.blocks = nn.ModuleList()
        in_dim = input_dim
        for h in hidden_dims:
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(in_dim, h),
                    nn.Tanh(),
                    nn.Linear(h, h),
                    nn.Tanh(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
            )
            in_dim = h
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return torch.sigmoid(self.head(x))

    # ------------------------------------------------------------------
    # scikit‑learn like interface
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        val_split: float = 0.2,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        early_stopping_patience: int = 20,
        verbose: bool = False,
    ) -> "QCNNHybridModel":
        """
        Train the model.

        Parameters
        ----------
        X, y : array‑like
            Training data and labels.
        val_split : float
            Proportion of data reserved for validation.
        epochs : int
            Maximum number of training epochs.
        batch_size : int
            Mini‑batch size.
        lr : float
            Optimiser learning rate.
        early_stopping_patience : int
            Number of epochs with no improvement to stop training.
        verbose : bool
            Print progress.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, stratify=y, random_state=42
        )
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = self(xb).squeeze()
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            val_loss = self._evaluate_loader(val_loader, criterion)
            if verbose:
                print(f"Epoch {epoch+1:03d} | Val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), "best_qcnn.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break

        self.load_state_dict(torch.load("best_qcnn.pt"))
        return self

    def _evaluate_loader(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                out = self(xb).squeeze()
                losses.append(criterion(out, yb).item())
        return np.mean(losses)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return class predictions (0/1)."""
        self.eval()
        with torch.no_grad():
            probs = self(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
        return (probs >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the positive class."""
        self.eval()
        with torch.no_grad():
            probs = self(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
        return probs.reshape(-1, 1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy."""
        return accuracy_score(y, self.predict(X))

    # ------------------------------------------------------------------
    # Hyper‑parameter search helper
    # ------------------------------------------------------------------
    @staticmethod
    def grid_search(
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, Iterable[Any]],
        cv: int = 3,
        verbose: int = 0,
    ) -> Tuple["QCNNHybridModel", Dict[str, Any]]:
        """
        Perform a simple grid search over the model’s hyper‑parameters.

        Parameters
        ----------
        X, y : array‑like
            Training data and labels.
        param_grid : dict
            Mapping of hyper‑parameter names to iterables of values.
        cv : int
            Number of cross‑validation folds.
        verbose : int
            Verbosity level.

        Returns
        -------
        best_estimator : QCNNHybridModel
            The model with the best cross‑validated performance.
        best_params : dict
            The hyper‑parameter configuration that yielded the best score.
        """
        best_score = -np.inf
        best_params = None
        best_model = None

        for lr in param_grid.get("lr", [1e-3]):
            for dropout in param_grid.get("dropout", [0.0]):
                for hidden in param_grid.get("hidden_dims", [[16, 16, 8, 4]]):
                    model = QCNNHybridModel(hidden_dims=hidden, dropout=dropout)
                    cv_scores = []
                    for fold in range(cv):
                        X_tr, X_val = X[fold * len(X) // cv : (fold + 1) * len(X) // cv], X[
                            (fold + 1) * len(X) // cv :
                        ]
                        y_tr, y_val = y[fold * len(y) // cv : (fold + 1) * len(y) // cv], y[
                            (fold + 1) * len(y) // cv :
                        ]
                        model.fit(X_tr, y_tr, epochs=200, lr=lr, early_stopping_patience=10, verbose=0)
                        score = model.score(X_val, y_val)
                        cv_scores.append(score)
                    mean_score = np.mean(cv_scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {"lr": lr, "dropout": dropout, "hidden_dims": hidden}
                        best_model = model
        return best_model, best_params


def QCNN() -> QCNNHybridModel:
    """Factory returning a default QCNNHybridModel instance."""
    return QCNNHybridModel()
