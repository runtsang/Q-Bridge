import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List, Callable, Optional
import numpy as np

class QuantumClassifierModel(nn.Module):
    """
    Classical feed‑forward classifier mirroring the quantum helper interface.
    The network architecture is fully configurable: number of hidden layers,
    hidden width, activation function, and optional dropout.  The class
    also exposes metadata (encoding, weight_sizes, observables) that
    matches the quantum version so that downstream pipelines can swap
    implementations without modification.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    hidden_dim : int, optional
        Width of each hidden layer (default: same as num_features).
    activation : Callable[[torch.Tensor], torch.Tensor], optional
        Activation function applied after each hidden layer.
    dropout : float, optional
        Dropout probability; 0.0 means no dropout.
    """
    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: Optional[int] = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or num_features
        self.encoding = list(range(num_features))
        layers: List[nn.Module] = []

        in_dim = num_features
        for _ in range(depth):
            lin = nn.Linear(in_dim, hidden_dim)
            layers.append(lin)
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.head = nn.Linear(in_dim, 2)
        layers.append(self.head)
        self.net = nn.Sequential(*layers)

        # store weight sizes for compatibility with quantum helper
        self.weight_sizes = [
            p.numel() for p in self.parameters()
        ]

        self.observables = [0, 1]  # placeholder for classification labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_metadata(self) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
        """
        Return the network, encoding indices, weight sizes and observables so
        that external code can inspect or clone the structure.
        """
        return self.net, self.encoding, self.weight_sizes, self.observables

    def train_one_epoch(
        self,
        dataloader,
        optimizer,
        criterion,
        device: torch.device = torch.device("cpu"),
    ) -> float:
        """
        Train the model for one epoch and return the average loss.
        """
        self.train()
        total_loss = 0.0
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = self(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader, device: torch.device = torch.device("cpu")) -> Tuple[float, float]:
        """
        Returns (accuracy, loss) on the provided dataloader.
        """
        self.eval()
        correct = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = self(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
        accuracy = correct / len(dataloader.dataset)
        return accuracy, total_loss / len(dataloader.dataset)

__all__ = ["QuantumClassifierModel"]
