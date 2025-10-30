"""Enhanced classical fully connected layer with advanced training utilities."""

from __future__ import annotations

from typing import Iterable, List, Optional, Union
import json
import torch
import torch.nn as nn
import numpy as np


class FullyConnectedLayer(nn.Module):
    """
    A versatile, drop‑in replacement for the original `FCL` wrapper.

    Parameters
    ----------
    input_dim : int, default 1
        Number of input features.
    layers : List[int] or None, default None
        Sizes of hidden layers. If None, a single linear layer mapping
        ``input_dim`` to 1 is created.
    dropout : float, default 0.0
        Dropout probability. 0 means no dropout.
    batch_norm : bool, default False
        Whether to apply batch normalization after each linear layer.
    """

    def __init__(
        self,
        input_dim: int = 1,
        layers: Optional[List[int]] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = [1]
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for out_dim in layers[:-1]:
            self.layers.append(nn.Linear(prev_dim, out_dim))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(out_dim))
            if dropout > 0.0:
                self.layers.append(nn.Dropout(dropout))
            prev_dim = out_dim
        self.layers.append(nn.Linear(prev_dim, layers[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                x = layer(x)
        return x

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward pass that accepts a list/array of parameters (thetas) representing input features.
        """
        with torch.no_grad():
            thetas_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            output = self.forward(thetas_tensor)
            return output.detach().cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience method for predicting on a batch of inputs.
        """
        with torch.no_grad():
            tensor = torch.as_tensor(X, dtype=torch.float32)
            out = self.forward(tensor)
            return out.detach().cpu().numpy()

    def save(self, path: str) -> None:
        """
        Save model parameters and architecture to a JSON file and state_dict.
        """
        torch.save(self.state_dict(), f"{path}.pt")
        meta = {
            "input_dim": self.layers[0].in_features,
            "layers": [layer.out_features for layer in self.layers if isinstance(layer, nn.Linear)],
            "batch_norm": any(isinstance(layer, nn.BatchNorm1d) for layer in self.layers),
            "dropout": next((layer.p for layer in self.layers if isinstance(layer, nn.Dropout)), 0.0),
        }
        with open(f"{path}.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "FullyConnectedLayer":
        """
        Load model from a saved checkpoint.
        """
        with open(f"{path}.json", "r") as f:
            meta = json.load(f)
        model = cls(
            input_dim=meta["input_dim"],
            layers=meta["layers"][1:],  # skip output layer
            dropout=meta.get("dropout", 0.0),
            batch_norm=meta.get("batch_norm", False),
        )
        model.load_state_dict(torch.load(f"{path}.pt"))
        return model


__all__ = ["FullyConnectedLayer"]
