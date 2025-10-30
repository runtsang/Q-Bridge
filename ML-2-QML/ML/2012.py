"""Hybrid classical classifier module with extended training utilities.

The module mirrors the original ``build_classifier_circuit`` signature but adds
drop‑out, skip connections, and a simple training helper.  The public API
remains compatible for backward‑compatibility users.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class QuantumClassifierModel:
    """Class that encapsulates a dense neural network and training utilities."""

    @staticmethod
    def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """
        Construct a dense network with dropout, skip connections, and a final linear head.

        Parameters
        ----------
        num_features : int
            Number of input features.
        depth : int
            Number of hidden layers.

        Returns
        -------
        Tuple[nn.Module, List[int], List[int], List[int]]
            The model, list of feature indices used for encoding, list of weight sizes
            per parameter, and list of observable class indices (here ``[0, 1]``).
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        for i in range(depth):
            # Linear layer
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            # Skip connection every two layers
            if i % 2 == 1:
                layers.append(nn.Linear(num_features, num_features))
            layers.append(nn.Dropout(p=0.2))
            in_dim = num_features

        # Output head
        head = nn.Linear(in_dim, 2)
        layers.append(head)

        net = nn.Sequential(*layers)

        encoding = list(range(num_features))
        weight_sizes = [p.numel() for p in net.parameters()]
        observables = [0, 1]  # class indices
        return net, encoding, weight_sizes, observables

    @staticmethod
    def train_classical(
        model: nn.Module,
        data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 10,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        """
        Very small training loop for the classical network.

        Parameters
        ----------
        model : nn.Module
            The model to train.
        data_loader : Iterable[Tuple[torch.Tensor, torch.Tensor]]
            Iterable yielding (inputs, targets) batches.
        epochs : int, optional
            Number of training epochs.
        lr : float, optional
            Learning rate for Adam optimiser.
        device : str, optional
            Device to run on (``'cpu'`` or ``'cuda'``).
        """
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()


# Backwards‑compatibility wrapper
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Delegate to ``QuantumClassifierModel.build_classifier_circuit``."""
    return QuantumClassifierModel.build_classifier_circuit(num_features, depth)


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
