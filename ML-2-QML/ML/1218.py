import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterable, List, Tuple, Optional
import numpy as np

class QuantumClassifierModel(nn.Module):
    """A flexible feed‑forward neural network that mirrors the quantum helper interface."""
    def __init__(
        self,
        num_features: int,
        hidden_sizes: List[int] = None,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        val_split: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> "QuantumClassifierModel":
        dataset = torch.utils.data.TensorDataset(X, y)
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state_dict = None

        for epoch in range(epochs):
            self.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = self(xb)
                    val_loss += criterion(logits, yb).item()
            val_loss /= len(val_loader)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | Val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state_dict = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
        return self

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            return preds

def build_classifier_circuit(
    num_features: int,
    hidden_sizes: List[int] = None,
    dropout: float = 0.0,
) -> Tuple[QuantumClassifierModel, List[int], List[int], List[int]]:
    model = QuantumClassifierModel(num_features, hidden_sizes, dropout)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables = [0]  # placeholder for compatibility
    return model, encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
