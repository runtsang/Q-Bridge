"""Classical classifier with residual connections and early stopping.

Provides a PyTorch Module that mirrors the interface of the quantum helper.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel(nn.Module):
    """
    A residual feed‑forward classifier that mimics the quantum circuit interface.
    The constructor builds a network of depth `depth` where each block consists
    of a Linear layer followed by ReLU.  Residual connections are added every
    two layers to ease gradient flow.  The class exposes the same metadata
    (encoding, weight_sizes, observables) that the quantum version returns.
    """
    def __init__(self, num_features: int, depth: int = 3, device: torch.device | str | None = None):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.device = torch.device(device or "cpu")
        self._build_network()

    def _build_network(self) -> None:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for i in range(self.depth):
            linear = nn.Linear(in_dim, self.num_features)
            layers.append(linear)
            layers.append(nn.ReLU(inplace=True))
            # Residual every second layer
            if i % 2 == 1:
                layers.append(nn.Identity())
            in_dim = self.num_features
        self.head = nn.Linear(in_dim, 2)
        self.network = nn.Sequential(*layers, self.head)
        self.to(self.device)
        # Metadata
        self.encoding = list(range(self.num_features))
        self.weight_sizes = [m.weight.numel() + m.bias.numel()
                             for m in self.network.modules()
                             if isinstance(m, nn.Linear)]
        self.observables = list(range(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.to(self.device))

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Static helper mirroring the quantum interface.
        Returns a freshly built model and its metadata.
        """
        model = QuantumClassifierModel(num_features, depth)
        return model.network, model.encoding, model.weight_sizes, model.observables

    def train_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        early_stop_patience: int = 10,
    ) -> List[float]:
        """
        Trains the network with Adam and early stopping.
        Returns the training loss history.
        """
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_loss = float("inf")
        patience = 0
        loss_history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)
            loss_history.append(epoch_loss)

            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    break
        return loss_history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns class indices for the input batch.
        """
        with torch.no_grad():
            logits = self.forward(X)
            return torch.argmax(logits, dim=1)
