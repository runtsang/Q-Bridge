import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, List

class QuantumClassifierModel(nn.Module):
    """
    Classical neural network that mirrors the interface of the quantum classifier.
    Supports configurable hidden layers, dropout and early stopping.
    """

    def __init__(self, num_features: int, hidden_layers: Optional[List[int]] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [num_features]
        layers = []
        in_dim = num_features
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, List[int], List[int], List[int]]:
        """
        Factory that returns a network and metadata analogous to the quantum version.
        """
        hidden = [num_features] * depth
        model = QuantumClassifierModel(num_features, hidden_layers=hidden, dropout=0.0)
        encoding = list(range(num_features))
        weight_sizes = [p.numel() for p in model.parameters()]
        observables = list(range(2))
        return model, encoding, weight_sizes, observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 64,
            lr: float = 1e-3, epochs: int = 50, early_stop: int = 5,
            verbose: bool = True) -> None:
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_loss = np.inf
        patience = 0
        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)
            if verbose:
                print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.4f}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
                torch.save(self.state_dict(), "best_model.pt")
            else:
                patience += 1
                if patience >= early_stop:
                    if verbose:
                        print("Early stopping triggered.")
                    break
        self.load_state_dict(torch.load("best_model.pt"))

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self(torch.from_numpy(X).float())
            probs = F.softmax(logits, dim=1)
        return probs.numpy()
