"""
EstimatorQNN – Classical feed‑forward regressor with training utilities.

This module extends the original tiny network by:
* Configurable depth, width, and dropout.
* Built‑in training loop with early stopping.
* Utility functions for synthetic data generation and evaluation.

The public API mirrors the original `EstimatorQNN()` factory, returning an instance ready for training.
"""

from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple, Dict, Any

__all__ = ["EstimatorQNN", "train_model", "evaluate_model", "make_synthetic_data"]


def EstimatorQNN(
    input_dim: int = 2,
    hidden_dims: Tuple[int,...] = (64, 32),
    dropout: float = 0.1,
    output_dim: int = 1,
    device: str | torch.device | None = None,
) -> nn.Module:
    """
    Create a configurable regression network.

    Parameters
    ----------
    input_dim : int
        Dimension of the input feature vector.
    hidden_dims : tuple[int,...]
        Sizes of hidden layers.
    dropout : float
        Dropout probability applied after each hidden layer.
    output_dim : int
        Dimension of the output (default 1 for scalar regression).
    device : str or torch.device, optional
        Target device for the model. Defaults to CUDA if available.

    Returns
    -------
    nn.Module
        Instantiated model moved to the specified device.
    """
    layers = []
    in_features = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_features, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_features = h
    layers.append(nn.Linear(in_features, output_dim))
    model = nn.Sequential(*layers)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 100,
    patience: int = 10,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Standard training loop with early stopping on validation loss.

    Parameters
    ----------
    model : nn.Module
        The network to train.
    dataloader : DataLoader
        DataLoader yielding (input, target) tuples.
    optimizer : optim.Optimizer
        Optimizer instance.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Device to perform computations on.
    epochs : int
        Maximum number of epochs.
    patience : int
        Number of epochs to wait for improvement before stopping.
    verbose : bool
        If True, prints epoch summaries.

    Returns
    -------
    dict
        Dictionary containing final loss and number of epochs trained.
    """
    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(dataloader.dataset)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.6f}")

        if epochs_no_improve >= patience:
            if verbose:
                print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"loss": best_loss, "epochs": epoch}


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Compute average loss over a dataset.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    dataloader : DataLoader
        DataLoader for evaluation data.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Target device.

    Returns
    -------
    float
        Mean loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)


def make_synthetic_data(
    num_samples: int = 1000,
    input_dim: int = 2,
    noise_std: float = 0.1,
    seed: int | None = None,
) -> TensorDataset:
    """
    Generate a toy regression dataset.

    The target is a smooth quadratic function of the inputs plus Gaussian noise.

    Parameters
    ----------
    num_samples : int
        Number of data points.
    input_dim : int
        Feature dimension.
    noise_std : float
        Standard deviation of Gaussian noise added to the target.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    TensorDataset
        Dataset ready for use with a DataLoader.
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    X = torch.randn(num_samples, input_dim, generator=rng)
    # Quadratic target: sum(x_i^2) + linear term
    y = (X**2).sum(dim=1, keepdim=True) + 0.5 * X.sum(dim=1, keepdim=True)
    y += noise_std * torch.randn_like(y, generator=rng)
    return TensorDataset(X, y)
