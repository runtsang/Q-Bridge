"""Hybrid classical regression with quantum kernel features."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data that mimics a superposition state.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Features array of shape (samples, num_features) and target array of shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """PyTorch dataset for the synthetic regression task."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumKernelRegressor(nn.Module):
    """Hybrid model that uses a quantum kernel as feature map followed by a
    classical ridge regression head.

    The quantum kernel is defined as the squared absolute overlap between
    product states obtained by applying RX rotations to each qubit.
    """

    def __init__(self, num_features: int, n_support: int = 100, lambda_reg: float = 1e-3):
        super().__init__()
        self.num_features = num_features
        self.n_support = n_support
        self.lambda_reg = lambda_reg
        # Support vectors are selected from the training set later.
        self.support_vectors = None
        self.head = nn.Linear(n_support, 1)

    def _encode_state(self, x: np.ndarray) -> np.ndarray:
        """Encode a single feature vector into a product state.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (num_features,).

        Returns
        -------
        np.ndarray
            State vector of shape (2**num_features,).
        """
        # Each qubit state: [cos(theta/2), -1j*sin(theta/2)]
        qubit_states = np.stack(
            [np.array([np.cos(t / 2), -1j * np.sin(t / 2)]) for t in x], axis=0
        )
        # Tensor product of qubit states
        state = qubit_states[0]
        for q in qubit_states[1:]:
            state = np.kron(state, q)
        return state

    def _kernel_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel matrix between X and the support set.

        Parameters
        ----------
        X : torch.Tensor
            Input batch of shape (batch_size, num_features).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (batch_size, n_support).
        """
        if self.support_vectors is None:
            raise RuntimeError("Support vectors not set. Call set_support_vectors first.")
        # Convert to numpy for encoding
        X_np = X.cpu().numpy()
        support_np = self.support_vectors.cpu().numpy()
        # Encode all samples
        psi_X = np.array([self._encode_state(x) for x in X_np])
        psi_S = np.array([self._encode_state(s) for s in support_np])
        # Compute overlap matrix
        overlap = np.abs(np.dot(psi_X, psi_S.conj().T)) ** 2
        return torch.from_numpy(overlap).float().to(X.device)

    def set_support_vectors(self, support: torch.Tensor):
        """Set the support vectors used for the kernel.

        Parameters
        ----------
        support : torch.Tensor
            Tensor of shape (n_support, num_features).
        """
        self.support_vectors = support

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Input batch of shape (batch_size, num_features).

        Returns
        -------
        torch.Tensor
            Predicted target of shape (batch_size,).
        """
        K = self._kernel_matrix(X)
        return self.head(K).squeeze(-1)

class QModel(QuantumKernelRegressor):
    """Alias for QuantumKernelRegressor for compatibility."""
    pass

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
