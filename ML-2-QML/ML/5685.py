"""
HybridClassifier for classical deep learning with residual connections and dropout.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class HybridClassifier(nn.Module):
    """
    A deep residual neural network with configurable depth, hidden sizes, and dropout.
    The network is designed for binary classification and exposes a familiar sklearn‑like API.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        num_features : int
            Dimensionality of the input features.
        hidden_dims : List[int]
            Sizes of successive hidden layers.
        dropout_rate : float
            Dropout probability applied after each hidden layer.
        lr : float
            Learning rate for the Adam optimizer.
        weight_decay : float
            L2 regularization strength.
        """
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_dims = hidden_dims

        layers: List[nn.Module] = []
        in_dim = num_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            # Residual connection only if dimensions match
            if in_dim == h_dim:
                layers.append(nn.Identity())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        verbose: bool = True,
    ) -> None:
        """
        Train the network with early stopping based on validation loss.

        Parameters
        ----------
        X_train : torch.Tensor
            Training data of shape (n_samples, num_features).
        y_train : torch.Tensor
            Training labels (0 or 1) of shape (n_samples,).
        X_val : Optional[torch.Tensor]
            Validation data.
        y_val : Optional[torch.Tensor]
            Validation labels.
        batch_size : int
            Batch size for training.
        epochs : int
            Maximum number of epochs.
        patience : int
            Number of epochs to wait for improvement before stopping.
        verbose : bool
            Whether to print progress.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_loss = float("inf")
        counter = 0

        for epoch in range(1, epochs + 1):
            self.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            if X_val is not None and y_val is not None:
                val_loss = self._evaluate(X_val, y_val, device, criterion)
                if verbose:
                    print(f"Epoch {epoch}: validation loss = {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                if counter >= patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch} completed.")

    def _evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
        criterion: nn.Module,
    ) -> float:
        self.eval()
        with torch.no_grad():
            logits = self.forward(X.to(device))
            loss = criterion(logits, y.to(device))
        return loss.item()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for input data.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, num_features).

        Returns
        -------
        torch.Tensor
            Predicted labels (0 or 1) of shape (n_samples,).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            preds = torch.argmax(logits, dim=1)
        return preds

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for input data.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, num_features).

        Returns
        -------
        torch.Tensor
            Probabilities of shape (n_samples, 2).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            probs = nn.functional.softmax(logits, dim=1)
        return probs

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
    ) -> Tuple[nn.Module, Iterable[int], List[int], Iterable[int]]:
        """
        Construct a feed‑forward classifier and metadata similar to the quantum variant,
        but enriched with dropout and residual connections.

        Parameters
        ----------
        num_features : int
            Number of input features.
        depth : int
            Number of hidden layers.

        Returns
        -------
        network : nn.Module
            The constructed network.
        encoding : Iterable[int]
            Indices of input features (identity mapping).
        weight_sizes : List[int]
            Number of trainable parameters per layer.
        observables : Iterable[int]
            Dummy observables for interface compatibility.
        """
        hidden_dims = [num_features] * depth
        classifier = HybridClassifier(num_features, hidden_dims, dropout_rate=0.2)
        weight_sizes = []
        for layer in classifier.network:
            if isinstance(layer, nn.Linear):
                weight_sizes.append(layer.weight.numel() + layer.bias.numel())
        encoding = list(range(num_features))
        observables = list(range(2))
        return classifier, encoding, weight_sizes, observables


__all__ = ["HybridClassifier"]
