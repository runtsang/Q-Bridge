"""PyTorch classification model utilities for the ML baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _ensure_float_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a float32 tensor on CPU."""
    if isinstance(array, torch.Tensor):
        tensor = array
    else:
        tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.dtype != torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def _ensure_long_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a long tensor on CPU."""
    if isinstance(array, torch.Tensor):
        tensor = array
    else:
        tensor = torch.as_tensor(array, dtype=torch.long)
    if tensor.dtype != torch.long:
        tensor = tensor.to(dtype=torch.long)
    return tensor


@dataclass
class TorchMLPClassifier:
    """A simple MLP classifier implemented with PyTorch."""

    n_features: int
    n_classes: int
    hidden_layer_sizes: Sequence[int] = (64, 32)
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 200
    batch_size: int = 32
    seed: int = 42
    device: torch.device | None = None
    history: List[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if len(self.hidden_layer_sizes) < 1:
            raise ValueError("hidden_layer_sizes must contain at least one hidden layer.")
        self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        layers: List[nn.Module] = []
        in_dim = self.n_features
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(p=float(self.dropout)))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.n_classes))

        self.model = nn.Sequential(*layers).to(self.device)
        print(self.model)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        verbose: bool = False,
        class_weights: np.ndarray | None = None,
    ) -> "TorchMLPClassifier":
        """Train the classifier on the provided dataset."""

        X_tensor = _ensure_float_tensor(np.asarray(X)).to(self.device)
        y_tensor = _ensure_long_tensor(np.asarray(y)).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fn = self.loss_fn

        self.model.train()
        self.history.clear()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            self.history.append(epoch_loss)
            if verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                print(f"    Epoch {epoch + 1:4d}/{self.epochs} - loss={epoch_loss:.6f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for the provided features."""
        X_tensor = _ensure_float_tensor(np.asarray(X)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels for the provided features."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def build_mlp_classifier(
    n_features: int,
    n_classes: int,
    *,
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    dropout: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 200,
    batch_size: int = 32,
    seed: int = 42,
    device: torch.device | None = None,
) -> TorchMLPClassifier:
    """Factory returning a :class:`TorchMLPClassifier` configured with the requested hyper-parameters."""

    return TorchMLPClassifier(
        n_features=n_features,
        n_classes=n_classes,
        hidden_layer_sizes=hidden_layer_sizes,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
