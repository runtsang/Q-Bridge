import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class QuantumClassifier(nn.Module):
    """
    Classical deep residual network that mirrors the interface of the quantum
    classifier.  It accepts the same metadata (encoding, weight_sizes,
    observables) and provides training, evaluation, and prediction methods.
    """

    def __init__(self, num_features: int, depth: int, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoding = list(range(num_features))
        self.weight_sizes = []
        layers = []
        in_dim = num_features
        for _ in range(depth):
            block = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_dim),
            )
            # Record parameter count of this block
            self.weight_sizes.append(
                sum(p.numel() for p in block.parameters())
            )
            layers.append(block)
            layers.append(nn.ReLU())
        self.res_blocks = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)
        self.weight_sizes.append(
            sum(p.numel() for p in self.head.parameters())
        )
        self.observables = [0, 1]  # indices of logits
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.res_blocks(x)
        logits = self.head(x)
        return logits

    def train_model(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    epochs: int = 10,
                    batch_size: int = 32,
                    lr: float = 1e-3,
                    device: str = "cpu") -> None:
        """
        Simple training loop using cross‑entropy loss.
        """
        dataset = TensorDataset(torch.from_numpy(X).float(),
                                torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.to(device)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)
            print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.4f}")

    def evaluate(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 batch_size: int = 64) -> float:
        """
        Return accuracy on the given data.
        """
        dataset = TensorDataset(torch.from_numpy(X).float(),
                                torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=batch_size)
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                logits = self.forward(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total

    def predict(self,
                X: np.ndarray,
                batch_size: int = 64) -> np.ndarray:
        """
        Return class probabilities for the input data.
        """
        dataset = TensorDataset(torch.from_numpy(X).float())
        loader = DataLoader(dataset, batch_size=batch_size)
        self.eval()
        probs = []
        with torch.no_grad():
            for xb, in loader:
                logits = self.forward(xb)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def save(self, path: str) -> None:
        """
        Persist the model state dict to disk.
        """
        torch.save({
            "model_state": self.state_dict(),
            "encoding": self.encoding,
            "weight_sizes": self.weight_sizes,
            "observables": self.observables,
        }, path)

    @classmethod
    def load(cls, path: str) -> "QuantumClassifier":
        """
        Load a previously saved model.
        """
        checkpoint = torch.load(path, map_location="cpu")
        # Estimate depth from weight_sizes (exclude head)
        depth = len(checkpoint["weight_sizes"]) - 2
        model = cls(num_features=len(checkpoint["encoding"]),
                    depth=depth)
        model.load_state_dict(checkpoint["model_state"])
        model.encoding = checkpoint["encoding"]
        model.weight_sizes = checkpoint["weight_sizes"]
        model.observables = checkpoint["observables"]
        return model
