"""Enhanced classical classifier mirroring the quantum helper interface.

The module extends the original simple feed‑forward classifier by adding
residual connections, batch‑normalisation and dropout.  The design allows
the network to be trained with higher depth without suffering from
vanishing gradients, while still exposing the same ``build_classifier_circuit`` API
as the quantum side for seamless hybrid experiments.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn

class ResidualClassifier(nn.Module):
    """Feed‑forward network with optional residual connections.

    Parameters
    ----------
    num_features : int
        Number of input features.
    depth : int
        Number of hidden layers.
    use_residual : bool, default True
        Whether to add a skip connection from input to each hidden block.
    dropout : float, default 0.0
        Dropout probability applied after each activation.
    batchnorm : bool, default True
        Whether to insert a BatchNorm1d after each linear layer.
    """
    def __init__(self, num_features: int, depth: int,
                 use_residual: bool = True,
                 dropout: float = 0.0,
                 batchnorm: bool = True) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.layers = nn.ModuleList()

        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            bn = nn.BatchNorm1d(num_features) if batchnorm else nn.Identity()
            self.layers.append(nn.ModuleDict({
                "linear": linear,
                "bn": bn,
                "relu": nn.ReLU(),
                "dropout": self.dropout_layer,
            }))
            in_dim = num_features

        self.head = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            y = layer["linear"](out)
            y = layer["bn"](y)
            y = layer["relu"](y)
            y = layer["dropout"](y)
            out = out + y if self.use_residual else y
        return self.head(out)

class QuantumClassifierModel:
    """Classical neural network that mimics the quantum helper interface.

    Attributes
        network (nn.Module): The constructed feature extractor.
        encoding (Iterable[int]): Indices of input features used for encoding.
        weight_sizes (List[int]): Size of each linear layer (weights+biases).
        observables (Iterable[int]): Dummy output labels for compatibility.
    """

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        *,
        use_residual: bool = True,
        dropout: float = 0.0,
        batchnorm: bool = True,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Construct a residual feed‑forward network with optional dropout and batch‑norm.

        Parameters
        ----------
        num_features: int
            Number of input features.
        depth: int
            Number of hidden layers.
        use_residual: bool, default True
            Whether to add a skip connection from input to each hidden block.
        dropout: float, default 0.0
            Dropout probability applied after each activation.
        batchnorm: bool, default True
            Whether to insert a BatchNorm1d after each linear layer.

        Returns
        -------
        Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]
            ``network``: the constructed nn.Sequential model.
            ``encoding``: list of feature indices used for encoding.
            ``weight_sizes``: list of parameter counts per layer.
            ``observables``: dummy observable indices (0 and 1 for binary classification).
        """
        network = ResidualClassifier(
            num_features=num_features,
            depth=depth,
            use_residual=use_residual,
            dropout=dropout,
            batchnorm=batchnorm,
        )

        # Count parameters for each linear layer
        weight_sizes: List[int] = []
        for m in network.layers:
            linear = m["linear"]
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        head = network.head
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        encoding = list(range(num_features))
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel"]
