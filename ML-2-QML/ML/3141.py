from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from QuantumKernelMethod import Kernel as QuantumKernel

class HybridKernelClassifier:
    """Hybrid kernel‑classifier that can operate in classical or quantum mode.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input data.
    depth : int, default 2
        Depth of the variational layers (used only in quantum mode).
    gamma : float, default 1.0
        RBF kernel width (used only in classical mode).
    use_quantum : bool, default False
        Whether to use the quantum kernel and classifier.
    """
    def __init__(self, num_features: int, depth: int = 2,
                 gamma: float = 1.0, use_quantum: bool = False) -> None:
        self.num_features = num_features
        self.depth = depth
        self.gamma = gamma
        self.use_quantum = use_quantum
        if use_quantum:
            self.kernel = QuantumKernel()
        else:
            # Classical RBF kernel implementation
            self.kernel = self._rbf_kernel
        self.classifier = LogisticRegression()
        self.X_train: np.ndarray | None = None

    def _rbf_kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a - b
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on training data."""
        self.X_train = X
        K = self._gram_matrix(X, X)
        self.classifier.fit(K, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new samples."""
        if self.X_train is None:
            raise RuntimeError("Model has not been fitted.")
        K = self._gram_matrix(X, self.X_train)
        return self.classifier.predict(K)

    def _gram_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix using the selected kernel."""
        gram = []
        for a in A:
            row = []
            for b in B:
                a_t = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
                b_t = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
                val = self.kernel(a_t, b_t).item()
                row.append(val)
            gram.append(row)
        return np.array(gram)
