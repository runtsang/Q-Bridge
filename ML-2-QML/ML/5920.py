"""QuantumClassifierModel: classical counterpart with advanced training utilities.

Provides a factory for a PyTorch feed‑forward network that mirrors the
quantum interface used by the QML module.  The network supports optional
batch normalisation, dropout and a configurable output head.  The returned
tuple contains the model, the list of input feature indices (the encoding),
the number of trainable parameters per layer and a list of observable
indices used for loss computation.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuantumClassifierModel"]


class QuantumClassifierModel:
    """Factory for a PyTorch feed‑forward classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input features.
    depth : int
        Number of hidden layers.
    hidden_dim : int, optional
        Width of each hidden layer (default: ``num_features``).
    dropout : float, optional
        Dropout probability for the hidden layers (default: 0.0).
    batch_norm : bool, optional
        Whether to insert a :class:`torch.nn.BatchNorm1d` after each
        linear layer (default: False).
    """

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """
        Construct a feed‑forward classifier and metadata similar to the quantum
        variant.

        Returns
        -------
        model : nn.Module
            Sequential model ready for training.
        encoding : List[int]
            Indices of input features used for encoding (identity mapping).
        weight_sizes : List[int]
            Number of trainable parameters in each layer (including bias).
        observables : List[int]
            Dummy observable list matching the output dimension.
        """
        hidden_dim = hidden_dim or num_features
        layers: List[nn.Module] = []

        # Input layer
        in_dim = num_features
        layers.append(nn.Linear(in_dim, hidden_dim))

        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        weight_sizes = [layers[0].weight.numel() + layers[0].bias.numel()]

        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            weight_sizes.append(layers[-4].weight.numel() + layers[-4].bias.numel())

        # Output head
        head = nn.Linear(hidden_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        model = nn.Sequential(*layers)

        # Metadata
        encoding = list(range(num_features))
        observables = list(range(2))  # placeholder for loss mapping

        return model, encoding, weight_sizes, observables

    @staticmethod
    def train(
        model: nn.Module,
        data_loader: Iterable,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        device: str = "cpu",
        early_stop_patience: int | None = None,
    ) -> Tuple[nn.Module, List[float]]:
        """
        Simple training routine with optional early stopping.

        Parameters
        ----------
        model : nn.Module
            The classifier to train.
        data_loader : Iterable
            Iterable yielding (inputs, targets).
        criterion : nn.Module
            Loss function.
        optimizer : torch.optim.Optimizer
            Optimiser.
        epochs : int, optional
            Number of epochs (default: 10).
        device : str, optional
            Device to use ('cpu' or 'cuda').
        early_stop_patience : int or None, optional
            Number of epochs with no improvement before stopping (default: None).

        Returns
        -------
        model : nn.Module
            Trained model.
        losses : List[float]
            History of training losses.
        """
        model.to(device)
        losses: List[float] = []
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for x, y in data_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)

            epoch_loss /= len(data_loader.dataset)
            losses.append(epoch_loss)

            if early_stop_patience is not None:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break

        return model, losses
