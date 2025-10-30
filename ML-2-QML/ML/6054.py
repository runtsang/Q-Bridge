"""QuantumClassifierModel implementation for classical neural networks.

The class mirrors the interface of the quantum counterpart, enabling
side‑by‑side experiments.  It builds a feed‑forward network with a
configurable depth, exposes the list of trainable parameters, and
provides training and inference utilities.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class QuantumClassifierModel:
    """Feed‑forward classifier with a user‑defined depth.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    depth : int
        Number of hidden layers.
    device : str | torch.device, default="cpu"
        Target device for tensors.
    """

    def __init__(self, num_features: int, depth: int, device: str | torch.device = "cpu"):
        self.num_features = num_features
        self.depth = depth
        self.device = torch.device(device)

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))  # binary classification
        self.network = nn.Sequential(*layers).to(self.device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Identity encoder – placeholder for feature engineering."""
        return x.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(self.encode(x))

    def train_model(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        verbose: bool = False,
    ) -> List[float]:
        """Train the model on a PyTorch DataLoader.

        Returns a list of training losses per epoch.
        """
        optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        self.network.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch, labels in train_loader:
                batch, labels = batch.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(batch)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(train_loader.dataset)
            losses.append(epoch_loss)
            if verbose:
                print(f"[Epoch {epoch+1}/{epochs}] loss={epoch_loss:.4f}")

        return losses

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions for a batch of inputs."""
        self.network.eval()
        with torch.no_grad():
            probs = torch.softmax(self.forward(x), dim=-1)
        return (probs[:, 1] >= threshold).long()

    def get_parameters(self) -> Iterable[torch.Tensor]:
        """Yield all trainable parameters for inspection."""
        return self.network.parameters()
