import torch
from torch import nn
from typing import Iterable, List, Tuple

class QCNNGen215(nn.Module):
    """
    Hybrid classical network that emulates the depth‑wise structure of a QCNN.

    The architecture is inspired by the original QCNNModel but replaces the
    single‑shot linear layers with a stack of small fully‑connected blocks
    that mirror the quantum convolution and pooling operations:

        * feature_map  –  encodes raw input into a higher dimensional space
        * conv layers –  two linear layers with non‑linearities
        * pool layers –  linear reduction of dimensionality
        * head        –  final classifier/regressor

    The network is intentionally lightweight so that it can be used as a
    baseline for comparing against the quantum version, or as a feature
    extractor in a hybrid pipeline.
    """

    def __init__(
        self,
        input_dim: int = 8,
        pool_sizes: Tuple[int,...] = (12, 8, 4),
        conv_sizes: Tuple[int,...] = (16, 12, 8, 4),
        head_units: int = 1,
        activation: nn.Module = nn.Tanh(),
        task: str = "classification",
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input feature vector.
        pool_sizes : Tuple[int,...]
            Sizes of the pooling (dimensionality reduction) layers.
        conv_sizes : Tuple[int,...]
            Sizes of the convolutional (expansion) layers.
        head_units : int
            Output dimension of the head. 1 for binary classification or
            regression; >1 for multi‑class/regression.
        activation : nn.Module
            Activation function used after each linear layer.
        task : str
            Either "classification" or "regression".  Determines the final
            activation (sigmoid vs. linear).
        """
        super().__init__()
        self.task = task
        self.activation = activation

        # Feature map – first expansion
        layers: List[nn.Module] = [nn.Linear(input_dim, conv_sizes[0]), activation]

        # Convolution + pooling blocks
        conv_iter = iter(conv_sizes[1:])
        pool_iter = iter(pool_sizes)
        for _ in range(len(pool_sizes)):
            # Convolution
            layers.append(nn.Linear(next(conv_iter), next(conv_iter)))
            layers.append(activation)
            # Pooling
            layers.append(nn.Linear(next(pool_iter), next(pool_iter)))
            layers.append(activation)

        # Final head
        layers.append(nn.Linear(next(conv_iter), head_units))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self.task == "classification":
            return torch.sigmoid(out)
        return out.squeeze(-1)

def QCNNGen215Classifier(input_dim: int = 8) -> QCNNGen215:
    """Factory for a binary classifier."""
    return QCNNGen215(
        input_dim=input_dim,
        pool_sizes=(12, 8, 4),
        conv_sizes=(16, 12, 8, 4),
        head_units=1,
        task="classification",
    )

def QCNNGen215Regressor(input_dim: int = 8) -> QCNNGen215:
    """Factory for a regression model."""
    return QCNNGen215(
        input_dim=input_dim,
        pool_sizes=(12, 8, 4),
        conv_sizes=(16, 12, 8, 4),
        head_units=1,
        task="regression",
    )

__all__ = [
    "QCNNGen215",
    "QCNNGen215Classifier",
    "QCNNGen215Regressor",
]
