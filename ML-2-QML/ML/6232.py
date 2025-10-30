"""
QuantumClassifierModel (classical) – PyTorch implementation with training, evaluation, and early stopping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterable, Tuple, List, Optional
import numpy as np


class QuantumClassifierModel:
    """
    A classical neural network that mimics the interface of the quantum classifier.
    It supports multi‑epoch training, early stopping, and flexible architecture design.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        hidden_dim: Optional[int] = None,
        lr: float = 1e-3,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        num_features : int
            Number of input features.
        depth : int, default 2
            Number of hidden layers.
        hidden_dim : int | None, default None
            Width of hidden layers; defaults to ``num_features``.
        lr : float, default 1e-3
            Learning rate for Adam optimizer.
        device : str, default "cpu"
            Torch device.
        seed : int | None, default None
            Random seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or num_features
        self.device = torch.device(device)
        self.lr = lr

        self.model = self._build_network()
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Metadata for reproducibility
        self.weight_sizes = self._compute_weight_sizes()
        self.encoding = list(range(num_features))
        self.observables = list(range(2))  # two‑class output

    def _build_network(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.hidden_dim, bias=True)
            layers.append(linear)
            layers.append(nn.ReLU())
            in_dim = self.hidden_dim
        # Final head
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def _compute_weight_sizes(self) -> List[int]:
        sizes = []
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                sizes.append(module.weight.numel() + module.bias.numel())
        return sizes

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 64,
        epochs: int = 200,
        patience: int = 20,
        verbose: bool = False,
    ) -> List[float]:
        """
        Train the network with early stopping.

        Returns a list of training losses per epoch.
        """
        X, y = X.to(self.device), y.to(self.device)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0
        losses: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            self.model.train()
            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)
            losses.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.4f}")

            # Early stopping
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                patience_counter = 0
                best_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break

        # Restore best parameters
        self.model.load_state_dict(best_state)
        return losses

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for input data.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device))
            return logits.argmax(dim=1)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """
        Return accuracy and cross‑entropy loss on a held‑out set.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device))
            loss = self.criterion(logits, y.to(self.device)).item()
            preds = logits.argmax(dim=1)
            acc = (preds == y.to(self.device)).float().mean().item()
        return acc, loss

    @property
    def metadata(self) -> dict:
        """
        Return model metadata: weight sizes, encoding, and observables.
        """
        return {
            "weight_sizes": self.weight_sizes,
            "encoding": self.encoding,
            "observables": self.observables,
        }

__all__ = ["QuantumClassifierModel"]
