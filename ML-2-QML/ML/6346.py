"""Classical implementation of a hybrid quantum‑kernel and classifier.

The class `QuantumKernelMethod` exposes both a radial‑basis‑function kernel
and a lightweight neural‑network classifier.  It mirrors the public API of
the original `QuantumKernelMethod.py` while extending it with a
trainable `nn.Module` that can be used as an alternative to the
pre‑computed‑kernel SVM.  The implementation is fully
PyTorch‑compatible and can be dropped into existing pipelines without
modifying downstream code.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def _rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute the RBF kernel matrix between two arrays."""
    diff = X[:, None, :] - Y[None, :, :]
    return np.exp(-gamma * np.sum(diff**2, axis=-1))


def _build_nn_classifier(num_features: int, depth: int, hidden: Iterable[int]) -> nn.Module:
    """Construct a simple feed‑forward network mirroring the quantum ansatz."""
    layers: list[nn.Module] = []
    in_dim = num_features
    for _ in range(depth):
        out_dim = hidden[0]
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)


class QuantumKernelMethod:
    """Hybrid classical kernel / neural‑network classifier.

    Parameters
    ----------
    gamma : float, default 1.0
        RBF kernel width.
    classifier : {'svm', 'nn'}, default'svm'
        Which classifier to train.  ``'svm'`` uses a pre‑computed‑kernel
        support‑vector machine.  ``'nn'`` builds a small PyTorch net.
    depth : int, default 3
        Depth of the neural network (used only for ``'nn'``).
    hidden : Iterable[int], default (64,)
        Size of hidden layers in the neural net.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        classifier: str = "svm",
        depth: int = 3,
        hidden: Iterable[int] | None = None,
    ) -> None:
        self.gamma = gamma
        self.classifier_type = classifier.lower()
        self.depth = depth
        self.hidden = hidden or (64,)
        self._X_train: np.ndarray | None = None
        self.model: nn.Module | SVC | None = None

    # ------------------------------------------------------------------
    # Kernel utilities
    # ------------------------------------------------------------------
    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """Return the RBF kernel matrix.

        If ``Y`` is ``None`` the matrix is computed against ``X`` itself.
        """
        Y = X if Y is None else Y
        return _rbf_kernel_matrix(X, Y, self.gamma)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the chosen classifier to ``X, y``."""
        self._X_train = X
        if self.classifier_type == "svm":
            K = self.kernel_matrix(X)
            self.model = SVC(kernel="precomputed")
            self.model.fit(K, y)
        elif self.classifier_type == "nn":
            device = torch.device("cpu")
            net = _build_nn_classifier(X.shape[1], self.depth, self.hidden).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
            X_t = torch.from_numpy(X.astype(np.float32)).to(device)
            y_t = torch.from_numpy(y.astype(np.int64)).to(device)

            net.train()
            for _ in range(200):
                optimizer.zero_grad()
                outputs = net(X_t)
                loss = criterion(outputs, y_t)
                loss.backward()
                optimizer.step()

            self.model = net
        else:
            raise ValueError(f"Unsupported classifier: {self.classifier_type}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions for ``X``."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self.classifier_type == "svm":
            K_test = self.kernel_matrix(X, self._X_train)  # type: ignore[arg-type]
            return self.model.predict(K_test)
        else:  # nn
            device = torch.device("cpu")
            net = self.model
            net.eval()
            with torch.no_grad():
                X_t = torch.from_numpy(X.astype(np.float32)).to(device)
                outputs = net(X_t)
                return outputs.argmax(dim=1).cpu().numpy()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy of the fitted model."""
        return accuracy_score(y, self.predict(X))


__all__ = ["QuantumKernelMethod"]
