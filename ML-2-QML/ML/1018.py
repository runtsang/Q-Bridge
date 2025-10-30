"""Hybrid classical‑quantum convolution module.

This module implements a two‑stage filter: a learnable classical kernel
computed from training data, followed by a quantum circuit that interprets
the weighted patch as a qubit state. The design keeps the public API
identical to the original `Conv` class so it can be dropped into existing
pipelines, and adds the ability to learn from data and to experiment with
different quantum backends or circuit depths.

Usage:

>>> from ConvEnhanced import ConvEnhanced
>>> conv = ConvEnhanced(kernel_size=3, backend="qasm_simulator", shots=200)
>>> conv.fit(X_train, y_train)          # learn kernel weights
>>> features = conv.transform(X_test)   # produce feature map
>>> probs = conv.predict(features)      # aggregate quantum outputs
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import qiskit
from ConvQuantum import QuanvCircuit

__all__ = ["ConvEnhanced"]

class ConvEnhanced(nn.Module):
    """
    A drop‑in replacement for the original Conv filter that learns a
    data‑driven kernel and then maps the weighted patch to a quantum circuit.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        backend: str | qiskit.providers.backend.Backend | None = None,
        shots: int = 200,
        threshold: float = 0.0,
        depth: int = 2,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the sliding window (e.g., 3×3).
        backend : str or Backend, optional
            Quantum backend; if None a local Aer simulator is used.
        shots : int
            Number of shots for the quantum measurement.
        threshold : float
            Threshold used to map input values to gate parameters.
        depth : int
            Depth of the random circuit added after the parameterized RX gates.
        lr : float
            Learning rate for the kernel learner (unused in this simple
            implementation but kept for API compatibility).
        epochs : int
            Number of training epochs (unused in this simple implementation).
        batch_size : int
            Batch size for training (unused in this simple implementation).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.depth = depth
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Learnable kernel weights (initialized to zeros)
        self.kernel_weights = nn.Parameter(torch.zeros(kernel_size ** 2))

        # Quantum filter
        self.quantum = QuanvCircuit(
            kernel_size=kernel_size,
            backend=backend,
            shots=shots,
            threshold=threshold,
            depth=depth,
        )

    def _patch_to_vector(self, patch: np.ndarray) -> np.ndarray:
        """Flatten a 2‑D patch to 1‑D vector."""
        return patch.reshape(-1)

    def _quantum_output(self, vector: np.ndarray) -> float:
        """Run the quantum filter on a weighted vector."""
        return self.quantum.run(vector)

    def forward(self, data: np.ndarray) -> float:
        """
        Compute the quantum output for a single image patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Quantum probability for the patch.
        """
        patch_vector = torch.as_tensor(data, dtype=torch.float32).view(-1)
        weighted = self.kernel_weights * patch_vector
        return self._quantum_output(weighted.detach().numpy())

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """
        Learn the kernel weights from training data.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples, kernel_size, kernel_size) containing
            training patches.
        y : np.ndarray, optional
            Target labels (0 or 1). If provided, the kernel is set to the
            difference between the mean of positive and negative patches;
            otherwise the kernel is set to the mean of all patches.
        """
        if y is None:
            # Unsupervised: mean of all patches
            mean_vec = torch.from_numpy(X.reshape(X.shape[0], -1).mean(axis=0))
            self.kernel_weights.data = mean_vec
        else:
            # Supervised: difference between positive and negative means
            pos_mask = y == 1
            neg_mask = y == 0
            pos_mean = torch.from_numpy(X[pos_mask].reshape(-1, self.kernel_size ** 2).mean(axis=0))
            neg_mean = torch.from_numpy(X[neg_mask].reshape(-1, self.kernel_size ** 2).mean(axis=0))
            self.kernel_weights.data = pos_mean - neg_mean

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Produce a feature map by sliding the filter over the input image.

        Parameters
        ----------
        X : np.ndarray
            3‑D array of shape (n_samples, H, W) or a single 2‑D image.

        Returns
        -------
        np.ndarray
            3‑D feature map of shape (n_samples, H - k + 1, W - k + 1).
        """
        if X.ndim == 2:
            X = X[np.newaxis,...]
        n_samples, H, W = X.shape
        out_h = H - self.kernel_size + 1
        out_w = W - self.kernel_size + 1
        features = np.zeros((n_samples, out_h, out_w), dtype=np.float32)

        for i in range(n_samples):
            for h in range(out_h):
                for w in range(out_w):
                    patch = X[i, h : h + self.kernel_size, w : w + self.kernel_size]
                    features[i, h, w] = self.forward(patch)
        return features

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Aggregate the quantum outputs over the feature map.

        Parameters
        ----------
        X : np.ndarray
            Feature map produced by `transform`.

        Returns
        -------
        np.ndarray
            1‑D array of aggregated probabilities (mean over spatial dims).
        """
        if X.ndim == 2:
            X = X[np.newaxis,...]
        return X.mean(axis=(1, 2))
