import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: Optional[int] = None
    device: str = "cpu"

class QuantumClassifierModel(nn.Module):
    """
    Classical feed‑forward classifier with optional residuals, batch‑norm and dropout.
    Provides convenient training, prediction and evaluation utilities.
    """
    def __init__(self,
                 num_features: int,
                 depth: int,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 device: str = "cpu",
                 seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        hidden_dim = hidden_dim or num_features
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, hidden_dim)
            layers.append(linear)
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            if in_dim == hidden_dim:
                layers.append(nn.Identity())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(self.device))

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor,
            *,
            val_split: float = 0.1,
            config: TrainingConfig = TrainingConfig()) -> Dict[str, List[float]]:
        """
        Train the model and return a history dictionary containing loss and accuracy
        for training and validation sets.
        """
        dataset = TensorDataset(X, y)
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_loader = DataLoader(dataset[:n_train],
                                  batch_size=config.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset[n_train:],
                                batch_size=config.batch_size)

        optimizer = optim.Adam(self.parameters(),
                               lr=config.lr,
                               weight_decay=config.weight_decay)
        criterion = nn.CrossEntropyLoss()

        history: Dict[str, List[float]] = {"train_loss": [],
                                           "val_loss": [],
                                           "train_acc": [],
                                           "val_acc": []}
        best_val = float("inf")
        patience = config.early_stopping_patience or 0
        wait = 0

        for epoch in range(config.epochs):
            self.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
            train_loss = epoch_loss / total
            train_acc = correct / total

            # Validation
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == yb).sum().item()
                    val_total += xb.size(0)
            val_loss /= val_total
            val_acc = val_correct / val_total

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                torch.save(self.state_dict(), "best_model.pt")
            else:
                wait += 1
                if patience and wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best
        self.load_state_dict(torch.load("best_model.pt"))
        return history

    def predict(self,
                X: torch.Tensor,
                *,
                probs: bool = False) -> torch.Tensor:
        """
        Return class indices or probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self(X.to(self.device))
            if probs:
                return F.softmax(logits, dim=1).cpu()
            return logits.argmax(dim=1).cpu()

    def evaluate(self,
                 X: torch.Tensor,
                 y: torch.Tensor) -> Dict[str, float]:
        """
        Compute accuracy and F1 score on the provided data.
        """
        preds = self.predict(X)
        acc = (preds == y).float().mean().item()
        tp = ((preds == 1) & (y == 1)).sum().item()
        fp = ((preds == 1) & (y == 0)).sum().item()
        fn = ((preds == 0) & (y == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"accuracy": acc, "f1": f1}

__all__ = ["QuantumClassifierModel", "TrainingConfig"]
