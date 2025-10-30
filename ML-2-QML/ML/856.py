"""Enhanced fully connected layer with configurable depth and dropout."""

import numpy as np
import torch
from torch import nn

class FCL(nn.Module):
    """
    Multi‑layer perceptron that mimics a quantum fully connected layer.

    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_layers : list[int] | None
        Sizes of hidden layers. If None, a single linear layer is used.
    dropout : float
        Dropout probability applied after each hidden layer.
    activation : str
        Activation function: 'tanh','relu', or'sigmoid'.
    """

    def __init__(
        self,
        n_features: int = 1,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        act = self._get_activation(activation)

        layers = [nn.Linear(n_features, hidden_layers[0] if hidden_layers else 1), act]
        if hidden_layers:
            for i in range(len(hidden_layers) - 1):
                layers.extend(
                    [
                        nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                        act,
                        nn.Dropout(dropout),
                    ]
                )
            layers.append(nn.Linear(hidden_layers[-1], 1))
        self.model = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(name: str):
        if name == "tanh":
            return nn.Tanh()
        if name == "relu":
            return nn.ReLU()
        if name == "sigmoid":
            return nn.Sigmoid()
        raise ValueError(f"Unsupported activation {name}")

    def run(self, thetas: np.ndarray | list[float]) -> np.ndarray:
        """
        Forward pass that interprets the input `thetas` as features for the network.

        Parameters
        ----------
        thetas : array-like
            Input features of shape (n_features,).

        Returns
        -------
        np.ndarray
            1‑D array containing the network's output.
        """
        input_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            out = self.model(input_tensor)
        return out.numpy().flatten()

__all__ = ["FCL"]
