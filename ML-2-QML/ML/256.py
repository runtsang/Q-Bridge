import torch
import torch.nn as nn
from typing import Iterable, List, Tuple


class QuantumClassifierModel:
    """
    Classical feed‑forward classifier with a residual‑style architecture.
    The interface mimics the quantum helper to keep downstream experiments
    consistent while adding dropout, batch‑norm and metadata accessors.
    """

    def __init__(self, num_features: int, depth: int, device: str = "cpu"):
        self.num_features = num_features
        self.depth = depth
        self.device = torch.device(device)
        self.model = self._build_model().to(self.device)
        self.encoding = list(range(num_features))
        self.weight_sizes = [p.numel() for p in self.model.parameters()]
        self.observables = [0, 1]  # placeholder for compatibility

    def _build_model(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, self.num_features))
            layers.append(nn.BatchNorm1d(self.num_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            in_dim = self.num_features
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 10,
        lr: float = 1e-3,
    ) -> None:
        """
        Train the model using CrossEntropyLoss and Adam.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        X = X.to(self.device)
        y = y.to(self.device)
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return class indices for the input batch.
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            logits = self.model(X)
            return torch.argmax(logits, dim=1).cpu()

    def get_metadata(self) -> Tuple[Iterable[int], List[int], List[int]]:
        """
        Return encoding indices, weight sizes and placeholder observables.
        """
        return self.encoding, self.weight_sizes, self.observables


__all__ = ["QuantumClassifierModel"]
