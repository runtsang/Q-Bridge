"""Hybrid classical classifier that optionally incorporates quantum embeddings."""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Optional

class QuantumClassifier(nn.Module):
    """
    A PyTorch classifier that can optionally prepend quantum feature embeddings.
    The architecture is a configurable depth feed‑forward network followed by a binary head.
    """
    def __init__(self, num_features: int, depth: int = 2, use_quantum: bool = False,
                 quantum_embedding: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.use_quantum = use_quantum
        self.quantum_embedding = quantum_embedding

        in_dim = num_features
        if self.use_quantum:
            # quantum embedding expands feature dimension by factor 2 (example)
            in_dim = num_features * 2

        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum and self.quantum_embedding is not None:
            qfeat = self.quantum_embedding(x)
            x = torch.cat([x, qfeat], dim=1)
        return self.network(x)

    def fit(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-3, epochs: int = 20,
            batch_size: int = 32) -> None:
        """
        Simple training loop using Adam optimizer and cross‑entropy loss.
        """
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            return logits.argmax(dim=1)

__all__ = ["QuantumClassifier"]
