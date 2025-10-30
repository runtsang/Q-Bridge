import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward network that mirrors the quantum helper but adds
    dropout, batch‑norm and a configurable hidden size.
    """
    layers = []
    hidden_dim = num_features * 2
    encoding = list(range(num_features))
    weight_sizes = []

    # Input block
    layers.append(nn.Linear(num_features, hidden_dim))
    layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(0.1))
    weight_sizes.append(layers[-4].weight.numel() + layers[-4].bias.numel())

    # Hidden layers
    for _ in range(depth - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        weight_sizes.append(layers[-4].weight.numel() + layers[-4].bias.numel())

    # Output head
    head = nn.Linear(hidden_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    net = nn.Sequential(*layers)
    observables = list(range(2))
    return net, encoding, weight_sizes, observables

class HybridQuantumClassifier:
    """
    Classical surrogate for the quantum classifier.  It provides a full training
    pipeline, data handling, and metrics while keeping the same metadata
    interface (encoding, weight sizes, observables) as the quantum version.
    """
    def __init__(self,
                 num_features: int,
                 depth: int = 3,
                 lr: float = 1e-3,
                 epochs: int = 200,
                 batch_size: int = 64,
                 device: str = "cpu"):
        self.net, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth)
        self.device = torch.device(device)
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.net.train()
        dataset = TensorDataset(X.to(self.device), y.to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.net(xb)
                loss = self.criterion(logits, yb.float())
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}  loss={epoch_loss / len(loader):.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            logits = self.net(X.to(self.device))
            probs = torch.sigmoid(logits)
            return (probs > 0.5).long()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        preds = self.predict(X)
        return (preds.squeeze() == y.to(self.device)).float().mean().item()

    def get_weight_sizes(self) -> Iterable[int]:
        return self.weight_sizes

    def get_encoding(self) -> Iterable[int]:
        return self.encoding

    def get_observables(self) -> list[int]:
        return self.observables
