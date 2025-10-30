import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """
    Classical classifier that mirrors the quantum helper interface.
    Supports configurable depth, optional residual connections, and dropout.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 3,
        use_residual: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        num_features : int
            Number of input features / qubits.
        depth : int
            Number of hidden layers.
        use_residual : bool
            If True, add skip connections from input to each hidden linear layer.
        dropout : float
            Dropout probability applied after each ReLU.
        bias : bool
            Whether to use bias in linear layers.
        seed : int
            Random seed for reproducibility.
        """
        torch.manual_seed(seed)
        self.num_features = num_features
        self.depth = depth
        self.use_residual = use_residual
        self.dropout = dropout

        layers: List[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(num_features, num_features, bias=bias))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier_head = nn.Linear(num_features, 2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_features).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, 2).
        """
        out = x
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                linear_out = layer(out)
                if self.use_residual:
                    out = out + linear_out
                else:
                    out = linear_out
            else:
                out = layer(out)
        return self.classifier_head(out)

    def build_classifier_circuit(self) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Mimic the quantum helper signature.

        Returns
        -------
        Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]
            The network, encoding indices, weight sizes, and observables.
        """
        encoding = list(range(self.num_features))
        weight_sizes = []
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                weight_sizes.append(layer.weight.numel() + layer.bias.numel())
        weight_sizes.append(self.classifier_head.weight.numel() + self.classifier_head.bias.numel())
        observables = list(range(2))
        return self, encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel"]
