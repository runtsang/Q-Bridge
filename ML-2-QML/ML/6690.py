import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Iterable, List

class QuantumClassifier:
    """
    Classical feed‑forward classifier that mirrors the interface of the quantum helper.
    Extends the original by providing training utilities, early‑stopping and device
    flexibility.
    """

    def __init__(self,
                 num_features: int,
                 depth: int = 2,
                 lr: float = 0.01,
                 epochs: int = 100,
                 batch_size: int = 32,
                 device: str = 'cpu'):
        self.num_features = num_features
        self.depth = depth
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.model, self.encoding, self.weight_sizes, self.observables = self.build_classifier_circuit(num_features, depth)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Construct a feed‑forward network with ReLU activations.
        Returns the network, an encoding mapping, weight sizes and observable indices.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

    def train(self, X: torch.Tensor, y: torch.Tensor, X_val: torch.Tensor | None = None,
              y_val: torch.Tensor | None = None) -> None:
        """
        Train the network using mini‑batch SGD with optional validation.
        """
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience, wait = 10, 0

        for epoch in range(self.epochs):
            self.model.train()
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()

            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val, compute_loss=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                if wait >= patience:
                    break

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return class predictions for input data.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device))
            return torch.argmax(logits, dim=1)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor, compute_loss: bool = False) -> float:
        """
        Return accuracy (and optionally loss) on a dataset.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device))
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y.to(self.device)).float().mean().item()
            if compute_loss:
                loss = self.criterion(logits, y.to(self.device)).item()
                return loss
            return acc

__all__ = ["QuantumClassifier"]
