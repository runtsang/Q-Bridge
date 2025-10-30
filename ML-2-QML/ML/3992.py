"""
QuantumHybridRegression.ml

An end‑to‑end classical regression/classification model that mirrors the quantum architecture
while providing a flexible head for both regression and binary classification.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

__all__ = ["QuantumHybridRegression", "RegressionDataset", "generate_superposition_data"]

# ------------------------------------------------------------
# Data generation utilities
# ------------------------------------------------------------
def generate_superposition_data(num_features: int,
                                num_samples: int,
                                noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature space.
    num_samples : int
        Number of samples to generate.
    noise_std : float, default=0.05
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    X : np.ndarray of shape (num_samples, num_features)
        Feature matrix.
    y : np.ndarray of shape (num_samples,)
        Target values.
    """
    X = np.random.uniform(-1.0, 1.0, size=(num_samples, num_features)).astype(np.float32)
    theta = X.sum(axis=1)
    y = np.sin(theta) + 0.1 * np.cos(2 * theta) + np.random.normal(0, noise_std, size=num_samples)
    return X, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that yields a dictionary with ``states`` and ``target``."""

    def __init__(self, num_samples: int, num_features: int):
        self.states, self.targets = generate_superposition_data(num_features, num_samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {"states": torch.tensor(self.states[idx], dtype=torch.float32),
                "target": torch.tensor(self.targets[idx], dtype=torch.float32)}


# --------------------------------------------------------------------
# Hybrid head for binary classification
# --------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """
    Differentiable sigmoid activation that mimics the quantum expectation head.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1.0 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """
    Dense head that uses a trainable shift before a sigmoid.
    """

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.linear(inputs)
        return HybridFunction.apply(logits, self.shift)


# --------------------------------------------------------------------
# Main model
# --------------------------------------------------------------------
class QuantumHybridRegression(nn.Module):
    """
    Classical regression/classification model that follows the same scaling
    strategy used in the quantum seed: a feature extractor followed by
    a hybrid head.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    hidden_dims : list[int], optional
        Sizes of the hidden layers in the classical feature extractor.
    regression : bool, default=True
        If True the model outputs a continuous value.
        If False the model outputs binary probabilities.
    shift : float, default=0.0
        Shift value used in the Hybrid head (binary classification only).
    """

    def __init__(self,
                 num_features: int,
                 hidden_dims: list[int] | None = None,
                 regression: bool = True,
                 shift: float = 0.0) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers)

        self.regression = regression
        if regression:
            self.head = nn.Linear(in_dim, 1)
        else:
            self.head = Hybrid(in_dim, shift=shift)

    def forward(self, states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        states : torch.Tensor of shape (batch, num_features)

        Returns
        -------
        torch.Tensor
            For regression: shape (batch,).
            For binary classification: shape (batch, 2) with probabilities.
        """
        features = self.feature_extractor(states)
        if self.regression:
            out = self.head(features).squeeze(-1)
            return out
        else:
            logits = self.head(features)
            probs = torch.sigmoid(logits).squeeze(-1)
            return torch.stack([probs, 1.0 - probs], dim=-1)

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper for inference.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(states)
