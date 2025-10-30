"""Hybrid classical regression dataset and model.

This module extends the original seed by adding a denoising head and a more
flexible dataset generator that can produce amplitude‑only, phase‑only,
or both superposition states.  The public API remains compatible with
the original, so downstream code can import ``RegressionDataset`` and
``generate_superposition_data`` unchanged.

The new ``QModel`` is a small MLP that first denoises the noisy
labels and then makes a final regression prediction.  It can be
trained separately or jointly with the quantum model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    mode: str = "both",
    noise_level: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic superposition states for regression.

    Parameters
    ----------
    num_features : int
        Number of features (equivalent to number of wires in the quantum model).
    samples : int
        Number of samples to generate.
    mode : str, optional
        One of ``"amplitude"``, ``"phase"``, or ``"both"``.  If ``"amplitude"``
        only the amplitude of the |0…0> component is used, if ``"phase"``
        only the phase of the |1…1> component is used, otherwise both are used.
    noise_level : float, optional
        Standard deviation of Gaussian noise added to the labels.

    Returns
    -------
    states : np.ndarray, shape (samples, num_features)
        Feature matrix.
    labels : np.ndarray, shape (samples,)
        Target values.
    """
    # Uniformly sample angles
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    # Build features
    if mode == "amplitude":
        states = np.cos(thetas)[:, None] * np.ones((samples, num_features))
    elif mode == "phase":
        states = np.exp(1j * phis)[:, None] * np.ones((samples, num_features))
    else:  # both
        states = np.cos(thetas)[:, None] * np.ones((samples, num_features)) + \
                 np.exp(1j * phis)[:, None] * np.ones((samples, num_features))

    # Labels: sin(2θ) * cos(φ) with optional noise
    labels = np.sin(2 * thetas) * np.cos(phis)
    labels += noise_level * np.random.randn(samples)

    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary with keys:
        - ``"states"``: torch.tensor of shape (num_features,)
        - ``"target"``: torch.tensor of shape ()
    """
    def __init__(self, samples: int, num_features: int, **kwargs):
        self.states, self.labels = generate_superposition_data(num_features, samples, **kwargs)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Classical MLP that first denoises the noisy labels and then makes a
    final regression prediction.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, denoise_dim: int = 32):
        super().__init__()
        self.denoiser = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, denoise_dim),
            nn.ReLU(),
            nn.Linear(denoise_dim, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor, shape (batch, num_features)
            Input states.

        Returns
        -------
        output : torch.Tensor, shape (batch,)
            Final regression output.
        """
        denoised = self.denoiser(state_batch).squeeze(-1)
        return self.head(denoised.unsqueeze(-1)).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
