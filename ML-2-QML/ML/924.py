import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """
    Classical feed‑forward classifier that mirrors the quantum helper interface.
    Supports multi‑class classification, data preprocessing, dropout, and L2 regularisation.
    """

    def __init__(self,
                 num_features: int,
                 hidden_dims: Iterable[int],
                 num_classes: int = 2,
                 dropout: float = 0.0,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 device: str = "cpu"):
        self.num_features = num_features
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device)

        self._build_network()
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)

    def _build_network(self):
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.num_classes))
        self.network = nn.Sequential(*layers).to(self.device)

    @staticmethod
    def build_classifier_circuit(num_features: int,
                                 depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Build a simple feed‑forward network mirroring the quantum helper.
        Returns the network, encoding indices, weight sizes per layer, and observable indices.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

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

    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: bool = False) -> None:
        """
        Train the network using mini‑batch SGD.
        """
        self.network.train()
        dataset = torch.utils.data.TensorDataset(X.to(self.device), y.to(self.device))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.network(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss / len(dataset):.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return class predictions.
        """
        self.network.eval()
        with torch.no_grad():
            logits = self.network(X.to(self.device))
            return torch.argmax(logits, dim=1)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.predict(X)

__all__ = ["QuantumClassifierModel"]
