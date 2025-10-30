"""Hybrid classical encoder and classifier for QuantumClassifierModel."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Iterable, List

class QuantumClassifierModel(nn.Module):
    """Classical encoder + classifier for the hybrid model."""
    def __init__(self, in_dim: int, latent_dim: int, hidden_dims: List[int] = None, device: str = "cpu"):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [in_dim // 2, latent_dim]
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 2)
        )
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits

    def fit(self, train_loader, epochs: int = 10, lr: float = 1e-3):
        """Train the encoder and classifier jointly."""
        self.train()
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(X.to(self.device))
            return torch.argmax(logits, dim=-1)
