import torch
import torch.nn as nn
from typing import List, Tuple

class QuantumClassifierModel:
    """Resilient classical classifier mirroring the quantum interface."""

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """
        Construct a residual feed‑forward network with optional dropout.

        Parameters
        ----------
        num_features : int
            Number of input features (also the hidden dimension).
        depth : int
            Number of stacked residual blocks.

        Returns
        -------
        network : nn.Module
            The constructed PyTorch model.
        encoding : List[int]
            Indices of input features (identical to range(num_features)).
        weight_sizes : List[int]
            Number of trainable parameters per linear layer.
        observables : List[int]
            Placeholder list of class indices for API compatibility.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            # Residual shortcut
            if in_dim == num_features:
                layers.append(nn.Identity())
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = [0, 1]  # 2-class classification
        return network, encoding, weight_sizes, observables
