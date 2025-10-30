"""Hybrid classical regression model with Bayesian linear regression head and early stopping.

This module extends the original seed by adding a Bayesian linear regression
head that outputs predictive mean and log‑variance, allowing uncertainty
estimation.  An EarlyStopping callback is also provided to halt training
once validation loss ceases to improve.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random superposition data.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples.

    Returns
    -------
    features : np.ndarray
        Input features of shape (samples, num_features).
    labels : np.ndarray
        Target values of shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch Dataset wrapping the superposition data.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class BayesianLinearHead(nn.Module):
    """
    Bayesian linear regression head that outputs predictive mean and log‑variance.
    """
    def __init__(self, in_features: int):
        super().__init__()
        # Means of weight and bias
        self.weight_mu = nn.Parameter(torch.randn(in_features, 1) * 0.1)
        self.bias_mu = nn.Parameter(torch.zeros(1))
        # Log‑variances (log σ²) of weight and bias
        self.weight_logvar = nn.Parameter(torch.full((in_features, 1), -5.0))
        self.bias_logvar = nn.Parameter(torch.full((1,), -5.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean and log‑variance of predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, in_features).

        Returns
        -------
        mean : torch.Tensor
            Predictive mean of shape (batch,).
        logvar : torch.Tensor
            Predictive log‑variance of shape (batch,).
        """
        mean = x @ self.weight_mu + self.bias_mu
        var = x.pow(2) @ torch.exp(self.weight_logvar) + torch.exp(self.bias_logvar)
        logvar = torch.log(var + 1e-8)  # add epsilon for numerical stability
        return mean.squeeze(-1), logvar.squeeze(-1)

class QuantumRegressionModel(nn.Module):
    """
    Hybrid classical regression model with a neural network backbone and Bayesian head.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.head = BayesianLinearHead(in_features=8)

    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing predictive mean and log‑variance.

        Parameters
        ----------
        state_batch : torch.Tensor
            Input batch of shape (batch, num_features).

        Returns
        -------
        mean : torch.Tensor
            Predictive mean of shape (batch,).
        logvar : torch.Tensor
            Predictive log‑variance of shape (batch,).
        """
        features = self.backbone(state_batch)
        return self.head(features)

    def loss(self, mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Negative log‑likelihood loss for Gaussian predictions.

        Parameters
        ----------
        mean : torch.Tensor
            Predictive mean.
        logvar : torch.Tensor
            Predictive log‑variance.
        target : torch.Tensor
            Ground‑truth targets.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss.
        """
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + ((target - mean).pow(2) / var))
        return nll.mean()

class EarlyStopping:
    """
    Early stopping utility to halt training when validation loss plateaus.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement after which training stops.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> None:
        """
        Update the early stopping status based on the latest validation loss.

        Parameters
        ----------
        val_loss : float
            Current validation loss.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

__all__ = [
    "QuantumRegressionModel",
    "RegressionDataset",
    "generate_superposition_data",
    "BayesianLinearHead",
    "EarlyStopping",
]
