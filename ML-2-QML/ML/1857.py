import torch
import torch.nn as nn
from typing import Iterable, Tuple, List, Callable

def build_classifier_circuit(num_features: int,
                             depth: int,
                             activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                             dropout_rate: float = 0.0,
                             batch_norm: bool = False) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward neural network that mirrors the interface of the quantum helper.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    activation : callable, optional
        Activation function to interleave between linear layers.
    dropout_rate : float, optional
        Dropout probability applied after every hidden layer.
    batch_norm : bool, optional
        If True, add a BatchNorm1d after each linear layer.

    Returns
    -------
    network : nn.Module
        Sequential model ready for training.
    encoding : Iterable[int]
        List of feature indices (identity mapping).
    weight_sizes : Iterable[int]
        Flat list of the number of parameters per linear layer.
    observables : List[int]
        Dummy observable identifiers matching the output classes.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    weight_sizes: List[int] = []

    # Build hidden layers
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        layers.append(linear)
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features))
        layers.append(activation)
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        in_dim = num_features

    # Output head
    head = nn.Linear(in_dim, 2)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    layers.append(head)

    network = nn.Sequential(*layers)

    encoding = list(range(num_features))
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit"]
