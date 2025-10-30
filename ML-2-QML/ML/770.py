"""QuantumClassifier: Classical feed‑forward classifier with advanced training utilities.

This class extends the original seed by providing multi‑class support, early‑stopping,
and an integrated‑gradients feature‑importance method.  The implementation
remains fully PyTorch‑based and can be dropped into any scikit‑learn pipeline.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QuantumClassifier(nn.Module):
    """
    A classical feed‑forward neural network that mimics the interface of the
    quantum helper while offering richer training facilities.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        num_classes: int = 2,
        hidden_dim: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.num_classes = num_classes
        self.device = device

        hidden = hidden_dim or num_features
        layers: List[nn.Module] = [nn.Linear(num_features, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, num_classes))
        self.network = nn.Sequential(*layers).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.to(self.device))

    def train_model(
        self,
        train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 20,
        lr: float = 1e-3,
        early_stop_patience: int = 5,
        verbose: bool = False,
    ) -> List[float]:
        """Train the network with Adam and optional early stopping."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_loss = float("inf")
        patience = 0
        losses: List[float] = []

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                logits = self.forward(x)
                loss = criterion(logits, y.to(self.device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
            losses.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    if verbose:
                        print(f"Early stopping after {epoch + 1} epochs.")
                    break
        return losses

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return class indices for the given inputs."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            return torch.argmax(logits, dim=1)

    def evaluate(
        self, loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """Compute accuracy over a data loader."""
        correct, total = 0, 0
        self.eval()
        with torch.no_grad():
            for x, y in loader:
                preds = self.predict(x)
                correct += (preds == y.to(self.device)).sum().item()
                total += y.size(0)
        return correct / total

    def integrated_gradients(
        self,
        x: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients for a single input.
        Returns a tensor of the same shape as *x* containing the attributions.
        """
        if baseline is None:
            baseline = torch.zeros_like(x).to(self.device)
        scaled_inputs = [
            baseline + (float(k) / steps) * (x - baseline) for k in range(steps + 1)
        ]
        grads: List[torch.Tensor] = []
        for inp in scaled_inputs:
            inp.requires_grad_(True)
            logits = self.forward(inp)
            loss = logits.sum()
            loss.backward()
            grads.append(inp.grad.detach().clone())
            inp.grad.zero_()
        avg_grads = torch.stack(grads, dim=0).mean(dim=0)
        integrated = (x - baseline) * avg_grads
        return integrated


__all__ = ["QuantumClassifier"]
